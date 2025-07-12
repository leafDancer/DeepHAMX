import sys
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import distrax
from typing import NamedTuple, Literal

import time
from omegaconf import OmegaConf
from pydantic import BaseModel

from envs import KSXEnv
from models import ActorCriticKS

class PPOConfig(BaseModel):
    env_name: Literal[
        "KS",
    ] = "KS"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 4096
    num_eval_envs: int = 512
    num_steps: int = 512
    total_timesteps: int = 400000000
    update_epochs: int = 40
    minibatch_size: int = 8192
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    act_std: float = 0.01
    vf_coef: float = 10
    max_grad_norm: float = 0.5
    save_model: bool = False

    class Config:
        extra = "forbid"

args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args)
env = KSXEnv()

num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size

def forward_fn(x):
    net = ActorCriticKS()
    logits, value = net(x)
    return logits, value

forward = hk.without_apply_rng(hk.transform(forward_fn))

optimizer = optax.chain(optax.clip_by_global_norm(
    args.max_grad_norm), optax.adam(args.lr, eps=1e-5))

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def make_update_fn():
    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        step_fn = jax.vmap(env.step)
        reset_fn = jax.vmap(env.reset)
        def _env_step(runner_state, unused):
            params, opt_state, env_state, last_obs, rng = runner_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            act_mean, value = forward.apply(params, last_obs)
            pi = distrax.ClippedNormal(loc=act_mean,scale=args.act_std,minimum=0,maximum=1)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = step_fn(env_state, action, keys)
            transition = Transition(
                env_state.terminated[...,0],
                action[...,0],
                value[...,0],
                jnp.squeeze(env_state.rewards)[...,0],
                log_prob[...,0],
                last_obs[...,0,:]
            )
            runner_state = (params, opt_state, env_state,
                            env_state.observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, args.num_steps
        )
        # 当我们没有auto reset时 建议在这里进行reset
        (params, opt_state, env_state, _, rng) = runner_state
        key,rng = jax.random.split(rng)
        keys = jax.random.split(key,args.num_envs)
        env_state = reset_fn(keys)
        runner_state = (params, opt_state, env_state,
                            env_state.observation, rng)
        # CALCULATE ADVANTAGE
        params, opt_state, env_state, last_obs, rng = runner_state
        _, last_val = forward.apply(params, last_obs)
        last_val = last_val[...,0]

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + args.gamma * next_value * (1 - done) - value
                gae = (
                    delta
                    + args.gamma * args.gae_lambda * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    act_mean, value = forward.apply(params, traj_batch.obs)
                    pi = distrax.ClippedNormal(loc=act_mean,scale=args.act_std,minimum=0,maximum=1)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-args.clip_eps, args.clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses,
                                          value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.clip_eps,
                            1.0 + args.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    total_loss = (
                        loss_actor
                        + args.vf_coef * value_loss
                    )
                    return total_loss, total_loss

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    params, traj_batch, advantages, targets)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            ###########
            
            ###########
            rng, _rng = jax.random.split(rng)
            batch_size = args.minibatch_size * num_minibatches
            assert (
                batch_size == args.num_steps * args.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state),  total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )
            update_state = (params, opt_state, traj_batch,
                            advantages, targets, rng)
            return update_state, total_loss

        update_state = (params, opt_state, traj_batch,
                        advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, args.update_epochs
        )
        params, opt_state, _, _, _, rng = update_state

        runner_state = (params, opt_state, env_state, last_obs, rng)
        return runner_state, loss_info
    return _update_step

@jax.jit
def evaluate(params, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.reset)(subkeys)
    R = jnp.zeros_like(state.rewards)
    discount = 1
    def cond_fn(tup):
        state, _, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key, discount = tup
        act_mean, value = forward.apply(params, state.observation)
        # action = logits.argmax(axis=-1)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, act_mean, keys)
        return state, R + state.rewards*discount, rng_key, discount * args.gamma
    state, R, _, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key, discount))
    # calculate discounted utility
    m_util = R.mean()
    k_end = state.k_cross.mean()
    return m_util, k_end

def train(rng):
    tt = 0
    st = time.time()
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, ) + env.observation_shape)
    params = forward.init(_rng, init_x)
    opt_state = optimizer.init(params=params)

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn()
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.reset))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (params, opt_state, env_state, env_state.observation, _rng)

    # warm up
    _, _ = jitted_update_step(runner_state)

    steps = 0

    # initial evaluation
    et = time.time()  # exclude evaluation time
    tt += et - st
    rng, _rng = jax.random.split(rng)
    m_util,k_end = evaluate(runner_state[0], _rng)
    print(f"num_updates={num_updates}")
    log = {"sec": tt, f"{args.env_name}/m_util": float(m_util),f"{args.env_name}/k_end": float(k_end), "steps": steps}
    print(log)

    st = time.time()
    for i in range(num_updates):
        runner_state, loss_info = jitted_update_step(runner_state)
        #print(f"average total loss = {loss_info[0].mean().item()}")
        steps += args.num_envs * args.num_steps
        # evaluation
        et = time.time()  # exclude evaluation time
        tt += et - st
        rng, _rng = jax.random.split(rng)
        m_util,k_end = evaluate(runner_state[0], _rng)
        log = {"iter":i, "sec": tt, f"{args.env_name}/m_util": float(m_util),f"{args.env_name}/k_end": float(k_end), "steps": steps}
        print(log)

        st = time.time()

    return runner_state

if __name__ == "__main__":
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng)