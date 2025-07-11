import jax
from jax import random
import jax.numpy as jnp 
import yaml
from functools import partial
from _src.struct import dataclass

def _get_details(self):
    self.prob_ag = jnp.zeros([2, 2])
    self.prob_trans = jnp.array(self.prob_trans)

    self.prob_ag = self.prob_ag.at[0, 0].set(self.prob_trans[0, 0] + self.prob_trans[0, 1])
    self.prob_ag = self.prob_ag.at[1, 1].set(self.prob_trans[3, 2] + self.prob_trans[3, 3])
    self.prob_ag = self.prob_ag.at[0, 1].set(1 - self.prob_ag[0, 0])
    self.prob_ag = self.prob_ag.at[1, 0].set(1 - self.prob_ag[1, 1])
    
    self.p_bb_uu = self.prob_trans[0, 0] / self.prob_ag[0, 0]
    self.p_bb_ue = 1 - self.p_bb_uu
    self.p_bb_ee = self.prob_trans[1, 1] / self.prob_ag[0, 0]
    self.p_bb_eu = 1 - self.p_bb_ee
    self.p_bg_uu = self.prob_trans[0, 2] / self.prob_ag[0, 1]
    self.p_bg_ue = 1 - self.p_bg_uu
    self.p_bg_ee = self.prob_trans[1, 3] / self.prob_ag[0, 1]
    self.p_bg_eu = 1 - self.p_bg_ee
    self.p_gb_uu = self.prob_trans[2, 0] / self.prob_ag[1, 0]
    self.p_gb_ue = 1 - self.p_gb_uu
    self.p_gb_ee = self.prob_trans[3, 1] / self.prob_ag[1, 0]
    self.p_gb_eu = 1 - self.p_gb_ee
    self.p_gg_uu = self.prob_trans[2, 2] / self.prob_ag[1, 1]
    self.p_gg_ue = 1 - self.p_gg_uu
    self.p_gg_ee = self.prob_trans[3, 3] / self.prob_ag[1, 1]
    self.p_gg_eu = 1 - self.p_gg_ee

    self.k_ss = ((1 / self.beta - (1 - self.delta)) / self.alpha) ** (1 / (self.alpha - 1))
    self.l_bar = 1.0 / 0.9   # time endowment normalizes labor supply to 1 in a bad state
    self.er_b = (1 - self.ur_b)  # employment rate in a bad aggregate state
    self.er_g = (1 - self.ur_g)  # employment rate in a good aggregate state

@dataclass
class StateKS:
    k_cross : jnp.ndarray
    ashock : jnp.int_
    ishock : jnp.ndarray


class KSXEnv:
    def __init__(self, cfg_path="cfg/KS.yaml") -> None:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        for k,v in cfg.items():
            setattr(self,k,v)
        _get_details(self)

    @partial(jax.jit, static_argnums=(0,))
    def _ashock2tfp(self, ashock:jnp.ndarray):
        '''convert ashock to tfp'''
        return (ashock * 2 - 1) * self.delta_a - 1
    
    @partial(jax.jit, static_argnums=(0,))
    def _state2obs(self, s:StateKS):
        obs = jnp.zeros((self.n_agent,4))
        obs = obs.at[...,0].set(s.k_cross)
        obs = obs.at[...,1].set(s.k_cross.mean())
        obs = obs.at[...,2].set(s.ashock)
        obs = obs.at[...,3].set(s.ishock)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key:random.PRNGKey):
        subkey, key = random.split(key)
        ashock = random.binomial(subkey,1,0.5)
        ur_rate = ashock * self.ur_g + (1 - ashock) * self.ur_b
        subkey, key = random.split(key)
        rand = random.uniform(subkey, self.n_agent)
        ishock = jnp.where(rand < ur_rate, jnp.zeros(self.n_agent), jnp.ones(self.n_agent))
        k_cross = jnp.ones(self.n_agent) * self.k_ss
        state = StateKS(k_cross,ashock,ishock)
        return state, self._state2obs(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key:random.PRNGKey, state:StateKS, actions:jnp.ndarray):
        '''
            state = {k_cross, ashock, ishock}
            actions.shape = (self.n_agent,1) and must be within (0,1)
        '''
        # Firstly, produce Y = tfp * K**alpha * emp**(1-alpha)
        k_mean = state.k_cross.mean()
        tfp = self._ashock2tfp(state.ashock) # total factor productivity
        er_real = state.ishock.mean() # real employment rate
        ur_real = 1 - er_real # real unemployment rate
        emp = self.l_bar * er_real # total labor supply
        tau = self.mu * ur_real / emp # labor tax rate
        R = 1 - self.delta + tfp * self.alpha * (k_mean / emp)**(self.alpha-1)
        wage =  tfp * (1-self.alpha) * (k_mean / emp)**self.alpha
        wealth = R * state.k_cross + (1-tau) * wage * self.l_bar * state.ishock + self.mu * wage * (1-state.ishock)
        # Secondly, consume 1-actions
        saving_rate = actions[...,0]
        csmp = jnp.clip(wealth * (1 - saving_rate), self.EPS, wealth-self.EPS)
        u = jnp.log(csmp)
        # Finally, update shocks
        subkey, key = random.split(key)
        if_keep = random.binomial(subkey, 1, 0.875)
        ashock = state.ashock * if_keep + (1 - state.ashock) * (1 - if_keep) # new ashock

        ur_rate = (1 - state.ashock) * (1 - ashock) * (1 - state.ishock) * self.p_bb_uu + (1 - state.ashock) * (1 - ashock) * state.ishock * self.p_bb_eu
        ur_rate += (1 - state.ashock) * ashock * (1 - state.ishock) * self.p_bg_uu + (1 - state.ashock) * ashock * state.ishock * self.p_bg_eu
        ur_rate += state.ashock * (1 - ashock) * (1 - state.ishock) * self.p_gb_uu + state.ashock * (1 - ashock) * state.ishock * self.p_gb_eu
        ur_rate += state.ashock * ashock * (1 - state.ishock) * self.p_gg_uu + state.ashock * ashock * state.ishock * self.p_gg_eu
        subkey,key = random.split(key)
        rand = random.uniform(subkey,self.n_agent)
        ishock = jnp.where(rand<ur_rate,jnp.zeros_like(state.ishock),state.ishock)
        state = state.replace(k_cross=wealth-csmp, ashock=ashock, ishock=ishock)

        return state, self._state2obs(state), u
        
if __name__ == "__main__":
    import time

    env = KSXEnv()
    subkey,key = random.split(random.PRNGKey(99))
    keys = random.split(subkey, (4,128))

    reset_fn = jax.pmap(jax.vmap(env.reset))
    s,o = reset_fn(keys)

    subkey,key = random.split(key)
    step_fn = jax.pmap(jax.vmap(env.step))
    keys = random.split(subkey, (4,128))
    rand_actions = random.uniform(key, (4,128,env.n_agent,1))
    st = time.time()
    for _ in range(1000):
        step_fn(keys, s, rand_actions)
    print(f"for loop 1000 steps: total {time.time()-st:.4f}s, {(time.time()-st)/1000:.5f}s per step.")
    
    
