import jax
import jax.numpy as jnp 
import util
import optax
import haiku as hk 
from functools import partial
from typing import NamedTuple
import pickle
from param import JNP_DTYPE

class Sample(NamedTuple):
    state: jnp.ndarray
    value: jnp.ndarray

class ValueTrainer():
    def __init__(self, config):
        self.config = config
        self.value_config = config["value_config"]
        d_in = config["n_basic"] + config["n_fm"] + config["n_gm"]
        self.n_device = 1
        def forward_fn(x): return util.FeedforwardModel(d_in, 1, self.value_config, name="v_net")(x)
        self.forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
        if config["n_gm"] > 0:
            # TODO generalize to multi-dimensional agt_s
            #self.gm_model = util.GeneralizedMomModel(1, config["n_gm"], config["gm_config"], name="v_gm")
            raise NotImplementedError
        self.train_vars = None
        self.optimizer = optax.adam(learning_rate=self.value_config["lr"],b1=0.99,b2=0.99)

        # 初始化参数 
        dummy_input = jnp.ones((256*50,4), dtype=JNP_DTYPE)
        self.model = self.forward.init(jax.random.PRNGKey(0), dummy_input)
        self.opt_state = self.optimizer.init(params=self.model[0])
        devices = jax.local_devices()[:self.n_device]
        self.model, self.opt_state = jax.device_put_replicated((self.model, self.opt_state), devices)

    def prepare_state_v(self, input_data, split_and_flat=True):
        if self.config["n_fm"] == 2:
            # 计算方差并扩展维度 (等价于 tf.reduce_variance + tile)
            k_var = jnp.var(input_data["agt_s"], axis=-2, keepdims=True)
            k_var = jnp.repeat(k_var, input_data["agt_s"].shape[-2], axis=-2)
            state = jnp.concatenate([input_data["basic_s"], k_var], axis=-1, dtype=JNP_DTYPE)
            
        elif self.config["n_fm"] == 0:
            # 选择特定列 (等价于 basic_s[..., 0:1] 和 basic_s[..., 2:])
            part1 = input_data["basic_s"][..., 0:1]
            part2 = input_data["basic_s"][..., 2:]
            state = jnp.concatenate([part1, part2], axis=-1, dtype=JNP_DTYPE)
            
        elif self.config["n_fm"] == 1:
            state = input_data["basic_s"]
        
        if self.config["n_gm"] > 0:
            raise NotImplementedError("n_gm > 0 尚未实现")
        
        if split_and_flat:
            valid_size = self.value_config['valid_size']*self.config['n_agt']
            samples = Sample(state=state,value=input_data["value"])
            samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)
            train_sample = jax.tree_util.tree_map(lambda x: x[valid_size:], samples)
            valid_sample = jax.tree_util.tree_map(lambda x: x[:valid_size], samples)
            return train_sample,valid_sample
        else:
            return state
    
    def train(self, key, num_epoch, batch_size, dataset): 
        train_sample, valid_sample = dataset
        valid_sample = jax.tree_util.tree_map(lambda x: x.reshape(self.n_device,-1,x.shape[-1]), valid_sample)
        train_step = jax.pmap(self.train_step,axis_name="i")
        loss_fn = jax.pmap(self.loss_fn)

        expand_factor = 2 # 整体训练规模的扩大系数 原过程训练规模太小
        real_num_ep = int(num_epoch * expand_factor * self.n_device / 2)
        print_freq = 20 * expand_factor
        for ep in range(real_num_ep):
            subkey,key = jax.random.split(key)
            ixs = jax.random.permutation(subkey, jnp.arange(train_sample.state.shape[0]))
            train_sample = jax.tree_util.tree_map(lambda x: x[ixs], train_sample)  # shuffle

            real_bs = batch_size * self.n_device * self.config['n_agt'] * expand_factor
            num_updates = train_sample.state.shape[0] // real_bs
            minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, self.n_device, -1) + x.shape[1:]), train_sample
            )

            for i in range(num_updates):
                minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
                self.model, self.opt_state = train_step(self.model, self.opt_state, minibatch)
        
            if (ep+1) % print_freq == 0:
                vloss,_ = loss_fn(self.model[0], self.model[1], valid_sample)
                vloss = vloss.mean()
                print(
                        "Real Epoch: %d, validation loss: %g" % (ep, vloss.item())
                        #"Value function learning epoch: %d" % (epoch)
                    )
               
    def train_step(self, model, opt_state, data):
        model_params, model_state = model
        grads, model_state = jax.grad(self.loss_fn, has_aux=True)(
            model_params, model_state, data
        )
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)
        model = (model_params, model_state)
        return model, opt_state


    def loss_fn(self, model_params, model_state, samples):
        pred_v, model_state = self.forward.apply(
            model_params, model_state, samples.state
        )
        value_loss = optax.l2_loss(pred_v, samples.value)
        value_loss = jnp.mean(value_loss)

        return value_loss, model_state
    
    def value_fn(self, model_param,model_state, input_data):
        state = self.prepare_state_v(input_data,split_and_flat=False)
        value,_ = self.forward.apply(model_param,model_state,state)
        return value
        
    def save_model(self, path="value_model.pkl"):
        with open(path, "wb") as f:
            dic = {
                "model_param": jax.device_get(self.model[0]),
                "model_state": jax.device_get(self.model[1]),
            }
            pickle.dump(dic, f)
        
    def load_model(self, path):
        with open(path, "wb") as f:
            dic = pickle.load(f)
        self.model = dic['model_param'],dic['model_state']


                
    