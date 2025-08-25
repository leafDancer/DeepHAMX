import os
import jax
import jax.numpy as jnp
import util
import simulation_KS as KS
import haiku as hk 
import optax
from tqdm import tqdm
from param import DTYPE,JNP_DTYPE

EPSILON = 1e-3

class PolicyTrainer():
    def __init__(self, vtrainers, init_ds, policy_path=None):
        self.config = init_ds.config
        self.policy_config = self.config["policy_config"]
        self.t_unroll = self.policy_config["t_unroll"]
        self.vtrainers = vtrainers
        self.valid_size = self.policy_config["valid_size"]
        self.sgm_scale = self.policy_config["sgm_scale"] # scaling param in sigmoid
        self.init_ds = init_ds
        self.value_sampling = self.config["dataset_config"]["value_sampling"]
        self.num_vnet = len(vtrainers)
        self.mparam = init_ds.mparam
        d_in = self.config["n_basic"] + self.config["n_fm"] + self.config["n_gm"]
        def forward_fn(x): return util.FeedforwardModel(d_in, 1, self.policy_config, name="p_net")(x)
        self.forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

        if self.config["n_gm"] > 0:
            # TODO generalize to multi-dimensional agt_s
            raise NotImplementedError
        
        self.train_vars = None
        if policy_path is not None:
            self.model.load_weights_after_init(policy_path)
            if self.config["n_gm"] > 0:
                raise NotImplementedError
            self.init_ds.load_stats(os.path.dirname(policy_path))
        self.discount = jnp.power(self.mparam.beta, jnp.arange(self.t_unroll, dtype=JNP_DTYPE))
        # to be generated in the child class
        self.policy_ds = None

        ####################################3
        lr_schedular = optax.schedules.exponential_decay(
            self.policy_config["lr_beg"],
            transition_steps=self.policy_config["num_step"],
            decay_rate=self.policy_config["lr_end"] / self.policy_config["lr_beg"],
            staircase=False
        )
        self.optimizer = optax.adam(learning_rate=lr_schedular,eps=1e-8,b1=0.99,b2=0.99)
        self.n_device = vtrainers[0].n_device
        dummy_input = jnp.ones((256*50,4), dtype=JNP_DTYPE)
        self.model = self.forward.init(jax.random.PRNGKey(0), dummy_input)
        self.opt_state = self.optimizer.init(params=self.model[0])
        devices = jax.local_devices()[:self.n_device]
        self.model, self.opt_state = jax.device_put_replicated((self.model, self.opt_state), devices)

        
    
    def prepare_state_p(self, input_data):
        if self.config["n_fm"] == 2:
            # Compute variance along the second last axis and keep dimensions
            k_var = jnp.var(input_data["agt_s"], axis=-2, keepdims=True)
            # Tile to match original shape
            k_var = jnp.tile(k_var, (1, input_data["agt_s"].shape[-2], 1))
            # Concatenate with basic_s
            state = jnp.concatenate([input_data["basic_s"], k_var], axis=-1, dtype=JNP_DTYPE)
        elif self.config["n_fm"] == 0:
            # Select all but the second column (index 1) from basic_s
            state = jnp.concatenate([input_data["basic_s"][..., 0:1], 
                                    input_data["basic_s"][..., 2:]], axis=-1, dtype=JNP_DTYPE)
        elif self.config["n_fm"] == 1:
            # Use basic_s as is
            state = input_data["basic_s"]
        
        if self.config["n_gm"] > 0:
            # Assuming gm_model is a JAX-compatible function
            raise NotImplementedError
        return state
    
    def policy_fn(self, model_param, model_state, input_data):
        state = self.prepare_state_p(input_data)
        #org = state.shape
        #state = state.reshape(self.n_device,-1,org[-1]) 
        p,_ = self.forward.apply(model_param,model_state,state)
        policy = jax.nn.sigmoid(p * self.sgm_scale)
        #policy = policy.reshape(*org[:-1],1)
        return policy

    def loss_fn(self, model_param, model_state, input_data, value_models):
        raise NotImplementedError

    def train_step(self,model,opt_state,data,value_models):
        model_params,model_state = model
        grads, (k_end, model_state) = jax.grad(self.loss_fn, has_aux=True)(
            model_params, model_state, data, value_models
        )
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)
        model = (model_params, model_state)
        return model, opt_state, k_end

    def train(self, key, num_step=None, batch_size=None):
        assert batch_size <= self.valid_size, "The valid size should be no smaller than batch_size."
        valid_data = dict((k, self.init_ds.datadict[k].astype(JNP_DTYPE)) for k in self.init_ds.keys)
        subkey,key = jax.random.split(key)
        ashock, ishock = self.simul_shocks(
            subkey, self.valid_size, self.t_unroll, self.mparam,
            state_init=self.init_ds.datadict
        )
        valid_data["ashock"] = ashock.astype(JNP_DTYPE)
        valid_data["ishock"] = ishock.astype(JNP_DTYPE)
        valid_data = jax.tree_util.tree_map(lambda x: x.reshape(self.n_device,-1,*x.shape[1:]), valid_data)

        freq_valid = self.policy_config["freq_valid"]
        n_epoch = num_step // freq_valid
        update_init = False
        train_step = jax.pmap(self.train_step, axis_name="i")
        loss_fn = jax.pmap(self.loss_fn)
        value_models = [vtr.model for vtr in self.vtrainers] 

        for n in range(n_epoch):
            for step in tqdm(range(freq_valid)):
                subkey,key = jax.random.split(key)
                train_data = self.sampler(subkey, batch_size, update_init)
                self.model,self.opt_state,k_end = train_step(self.model,self.opt_state,train_data,tuple(value_models))
                n_step = n*freq_valid + step
                if self.value_sampling != "bchmk" and n_step % self.policy_config["freq_update_v"] == 0 and n_step > 0:
                    update_init = self.policy_config["update_init"]
                    vds = self.get_valuedataset(update_init)
                    vds = self.vtrainers[0].prepare_state_v(vds)
                    for vtr in self.vtrainers:
                        subkey, key = jax.random.split(key)
                        vtr.train(
                            subkey,
                            self.config["value_config"]["num_epoch"],
                            self.config["value_config"]["batch_size"],
                            vds
                        )
                    value_models = [vtr.model for vtr in self.vtrainers] 

            m_util,_ = loss_fn(self.model[0], self.model[1], valid_data, value_models)

            print(
                "Step: %d, valid util: %g, k_end: %g" %
                (freq_valid*(n+1), -m_util.mean().item(), k_end.mean().item())
            )

    def simul_shocks(self, key, n_sample, T, mparam, state_init):
        raise NotImplementedError
    
    def sampler(self, key, batch_size, update_init=False):
        train_data = self.policy_ds.next_batch(batch_size)
        ashock, ishock = self.simul_shocks(key, batch_size, self.t_unroll, self.mparam, train_data)
        train_data["ashock"] = ashock.astype(JNP_DTYPE)
        train_data["ishock"] = ishock.astype(JNP_DTYPE)
        # TODO test the effect of epoch_resample
        if self.policy_ds.epoch_used > self.policy_config["epoch_resample"]:
            self.update_policydataset(update_init)
        
        train_data = jax.tree_util.tree_map(lambda x: x.reshape(self.n_device, -1, *x.shape[1:]),train_data)
        return train_data

    def update_policydataset(self, update_init=False):
        raise NotImplementedError

    def get_valuedataset(self, update_init=False):
        raise NotImplementedError


class KSPolicyTrainer(PolicyTrainer):
    def __init__(self, key, vtrainers, init_ds, policy_path=None):
        super().__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.k_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        subkey, self.key = jax.random.split(key)
        self.policy_ds = self.init_ds.get_policydataset(subkey, init_policy, policy_type, update_init=False)
    
    def update_policydataset(self, update_init=False):
        subkey, self.key = jax.random.split(self.key)
        model = jax.tree_util.tree_map(lambda x:x[0],self.model)
        self.policy_ds = self.init_ds.get_policydataset(subkey, self.current_c_policy, "nn_share", update_init, model)

    def get_valuedataset(self, update_init=False):
        model = jax.tree_util.tree_map(lambda x:x[0],self.model)
        return self.init_ds.get_valuedataset(self.current_c_policy, "nn_share", update_init, model)

    def simul_shocks(self, key, n_sample, T, mparam, state_init):
        return KS.simul_shocks(key, n_sample, T, mparam, state_init)   
    
    def current_c_policy(self, model_param, model_state, k_cross, ashock, ishock):
        k_mean = jnp.mean(k_cross, axis=1, keepdims=True)
        k_mean = jnp.repeat(k_mean, self.mparam.n_agt, axis=1)
        ashock = jnp.repeat(ashock, self.mparam.n_agt, axis=1)
        basic_s = jnp.stack([k_cross, k_mean, ashock, ishock], axis=-1, dtype=JNP_DTYPE)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s")

        full_state_dict = {
            "basic_s": basic_s
        }
        agt_s = self.init_ds.normalize_data(k_cross[:, :, None], key="agt_s")
        full_state_dict['agt_s'] = agt_s
        c_share = self.policy_fn(model_param, model_state, full_state_dict)[...,0]
        return c_share

    def loss_fn(self, model_param, model_state, input_data, value_models):
        k_cross  = input_data["k_cross"]           # (n_path, n_agt)
        ashock   = input_data["ashock"]            # (n_path, T)
        ishock   = input_data["ishock"]            # (n_path, n_agt, T)

        v_params, v_states = zip(*value_models)

        # function of simulate one path
        one = lambda k0, a_p, i_p: self._one_path_loss(
                    model_param, model_state,
                    k0, a_p, i_p,
                    v_params, v_states)

        # paralelly simulate n path using jax.vmap
        loss_batch, k_end_batch = jax.vmap(one)(k_cross, ashock, ishock)

        return loss_batch.mean(), (k_end_batch.mean(), model_state)
    
    def _one_path_loss(self, model_param, model_state,  # ← NO vmap
                    k0, ashock_path, ishock_path,    # (n_agt,), (T,), (n_agt,T)
                    v_params, v_states               # list(zip(param,state))
                    ):
        """A single path (n_agt,) → Scalar loss, End-simulation capital"""

        # ▶ prepare value network
        def batched_value(full_state_dict):
            """Return (n_agt,) value"""
            def _single(model, state):
                return self.vtrainers[0].value_fn(model, state, full_state_dict)[..., 0]
            vals = jnp.stack([_single(p, s) for p, s in zip(v_params, v_states)], axis=0)
            # mean over ensemble
            return self.init_ds.unnormalize_data(vals.mean(0), key="value")   # (n_agt,)

        discount = self.discount                    # (T,)
        n_agt     = self.mparam.n_agt
        opt_game  = self.policy_config["opt_type"] == "game"

        def step(carry, inp):
            util_sum, k_cross = carry               # (n_agt,), (n_agt,)

            a_t, i_t, disc_t, t_idx = inp           # scalar, (n_agt,), scalar, int32

            k_mean = k_cross.mean(0, keepdims=True)         # (1,)
            # ---------- prepare network input ----------
            basic_s = jnp.stack([k_cross,             # (n_agt,)
                                jnp.repeat(k_mean, n_agt),
                                jnp.repeat(a_t,   n_agt),
                                i_t], axis=-1, dtype=JNP_DTYPE)                  # (n_agt,4)
            agt_s   = k_cross[:,None]                            # (n_agt,1)

            full_state = {
                "basic_s": self.init_ds.normalize_data(basic_s, key="basic_s"),
                "agt_s"  : self.init_ds.normalize_data(agt_s  , key="agt_s"),
            }

            # ---------- Last Iteration of Simulation ----------
            def last_step(_):
                v = batched_value(full_state)        # (n_agt,)
                return util_sum + disc_t * v, k_cross

            # ---------- Normal Iteration of Simulation ----------
            def normal_step(_):
                c_share = self.policy_fn(model_param, model_state, full_state)[..., 0]  # (n_agt,)
                if opt_game:                        # Only optimize agent 0
                    c_share = jnp.concatenate([c_share[:1], jax.lax.stop_gradient(c_share[1:])], axis=0, dtype=JNP_DTYPE)

                tau  = jnp.where(a_t < 1, self.mparam.tau_b, self.mparam.tau_g)
                emp  = jnp.where(a_t < 1,
                                self.mparam.l_bar * self.mparam.er_b,
                                self.mparam.l_bar * self.mparam.er_g)

                R     = 1 - self.mparam.delta + a_t * self.mparam.alpha * (k_mean/emp)**(self.mparam.alpha-1)
                wage  = a_t * (1 - self.mparam.alpha) * (k_mean/emp)**self.mparam.alpha

                wealth = R * k_cross \
                    + (1 - tau) * wage * self.mparam.l_bar * i_t \
                    + self.mparam.mu * wage * (1 - i_t)

                csmp   = jnp.clip(c_share * wealth, EPSILON, wealth - EPSILON)
                k_next = wealth - csmp
                util   = jnp.log(csmp)

                return util_sum + disc_t * util, k_next

            util_sum, k_cross = jax.lax.cond(t_idx == (discount.shape[0] - 1),
                                        last_step, normal_step, operand=None)
            return (util_sum, k_cross), None   # no scan output

        # ---------- scan over time ----------
        T = ashock_path.shape[0]
        scan_inputs = (ashock_path,
                    ishock_path.T,             # (T,n_agt) → (n_agt,) per step
                    discount,
                    jnp.arange(T))             # step index

        (util_fin, k_fin), _ = jax.lax.scan(step,              # carry
                                        (jnp.zeros(n_agt, DTYPE), k0),
                                        scan_inputs,
                                        length=T)

        # ---------- Aggregate ----------
        if self.policy_config["opt_type"] == "socialplanner":
            m_util = -jnp.mean(util_fin)
        else:                                       # game → agent 0
            m_util = -util_fin[0]
        return m_util, k_fin.mean()                 # scalar, scalar