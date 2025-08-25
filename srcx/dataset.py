import jax
import jax.numpy as jnp 
import os
import scipy.io as sio
import simulation_KS as KS
import json 
from functools import partial
from param import DTYPE,JNP_DTYPE

EPSILON = 1e-3


@partial(jax.jit, static_argnums=(2,)) 
def _jit_shuffle(datadict, key, size):
    ixs = jax.random.permutation(key, jnp.arange(size))
    return jax.tree_util.tree_map(lambda x: x[ixs], datadict)

class JNumpyEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, jnp.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

class BasicDataSet():
    def __init__(self, key, datadict=None):
        self.key = key
        self.datadict, self.keys = None, None
        self.size, self.idx_in_epoch, self.epoch_used = None, None, None
        if datadict:
            subkey, self.key = jax.random.split(self.key)
            self.update_datadict(subkey, datadict)

    def update_datadict(self, key, datadict):
        self.datadict = datadict
        self.keys = datadict.keys()
        size_list = [datadict[k].shape[0] for k in self.keys]
        for i in range(1, len(size_list)):
            assert size_list[i] == size_list[0], "The size does not match."
        self.size = size_list[0]
        self.shuffle(key)
        self.epoch_used = 0

    def shuffle(self,key):
        self.datadict = _jit_shuffle(self.datadict, key, self.size)
        self.idx_in_epoch = 0

    def next_batch(self,batch_size):
        if self.idx_in_epoch + batch_size > self.size:
            subkey, self.key = jax.random.split(self.key)
            self.shuffle(subkey)
            self.epoch_used += 1
        idx = slice(self.idx_in_epoch, self.idx_in_epoch+batch_size)
        self.idx_in_epoch += batch_size
        return dict((k, self.datadict[k][idx]) for k in self.keys)

class DataSetwithStats(BasicDataSet):
    def __init__(self, key, stats_keys, datadict=None):
        subkey, self.key = jax.random.split(key)
        super().__init__(subkey,datadict)
        self.stats_keys = stats_keys
        self.stats_dict, self.stats_dict_tf = {}, {}
        for k in stats_keys:
            self.stats_dict[k] = None
            self.stats_dict_tf[k] = None

    def update_stats(self, data, key, ma):
        # data can be of shape B * d or B * n_agt * d
        axis_for_mean = tuple(list(range(len(data.shape)-1)))
        if self.stats_dict[key] is None:
            mean, std = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
        else:
            mean_new, std_new = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
            mean, std = self.stats_dict[key]
            mean = mean * ma + mean_new * (1-ma)
            std = std * ma + std_new * (1-ma)
        self.stats_dict[key] = (mean, std)

    def normalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
        else:
            mean, std = self.stats_dict[key]
        return (data - mean) / std

    def unnormalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
        else:
            mean, std = self.stats_dict[key]
        return data * std + mean

    def save_stats(self, path):
        with open(os.path.join(path, "stats.json"), "w") as fp:
            json.dump(self.stats_dict, fp, cls=JNumpyEncoder)

    def load_stats(self, path):
        with open(os.path.join(path, "stats.json"), "r") as fp:
            saved_stats = json.load(fp)
        for key in saved_stats:
            assert key in self.stats_dict, "The key of stats_dict does not match!"
            mean, std = saved_stats[key]
            mean, std = jnp.asarray(mean).astype(JNP_DTYPE), jnp.asarray(std).astype(JNP_DTYPE)
            self.stats_dict[key] = (mean, std)


class InitDataSet(DataSetwithStats):
    def __init__(self, key, mparam, config):
        subkey, self.key = jax.random.split(key)
        super().__init__(subkey,stats_keys=["basic_s", "agt_s", "value"])
        self.mparam = mparam
        self.config = config
        self.n_basic = config["n_basic"]
        self.n_fm = config["n_fm"]  # fixed moments
        self.n_path = config["dataset_config"]["n_path"]
        self.t_burn = config["dataset_config"]["t_burn"]
        self.c_policy_const_share = lambda *args: config["init_const_share"]
        if not config["init_with_bchmk"]:
            assert config["policy_config"]["update_init"], \
                "Must update init data during learning if bchmk policy is not used for sampling init"

    def update_with_burn(self, policy, policy_type, t_burn=None, state_init=None):
        if t_burn is None:
            t_burn = self.t_burn
        if state_init is None:
            state_init = self.datadict
        simul_data = self.simul_k_func(
            self.n_path, t_burn, self.mparam,
            policy, policy_type, state_init=state_init
        )
        subkey, self.key = jax.random.split(self.key)
        self.update_from_simul(subkey, simul_data)

    def update_from_simul(self, key, simul_data):
        init_datadict = dict((k, simul_data[k][..., -1].copy()) for k in self.keys)
        for k in self.keys:
            if len(init_datadict[k].shape) == 1:
                init_datadict[k] = init_datadict[k][:, None] # for macro init state like N in JFV
        notnan = ~(jnp.isnan(init_datadict["k_cross"]).any(axis=1))
        
        num_nan = jnp.sum(~notnan)
    
        def _raise_error():
            raise ValueError(f"Training Stop: There are {num_nan} NaN Samples!")
        
        # check nans
        jax.lax.cond(
            num_nan > 0,
            lambda: jax.debug.callback(_raise_error),
            lambda: None
        )
        self.update_datadict(key, init_datadict)

    def process_vdatadict(self, v_datadict):
        idx_nan = jnp.logical_or(
            jnp.isnan(v_datadict["basic_s"]).any(axis=(1, 2)),
            jnp.isnan(v_datadict["value"]).any(axis=(1, 2))
        )
        ma = self.config["dataset_config"]["moving_average"]
        for key, array in v_datadict.items():
            array = array[~idx_nan].astype(JNP_DTYPE)
            self.update_stats(array, key, ma)
            v_datadict[key] = self.normalize_data(array, key)
        print("Average of total utility %f." % (self.stats_dict["value"][0][0]))

        valid_size = self.config["value_config"]["valid_size"]
        n_sample = v_datadict["value"].shape[0]
        if valid_size > 0.2*n_sample:
            valid_size = int(0.2*n_sample)
            print("Valid size is reduced to %d according to small data size!" % valid_size)
        print("The dataset has %d samples in total." % n_sample)
        return v_datadict

    def get_policydataset(self, key, policy, policy_type, update_init=False, model=None):
        policy_config = self.config["policy_config"]
        simul_data = self.simul_k_func(
            self.n_path, policy_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict, model=model
        )
        if update_init:
            subkey,key = jax.random.split(key)
            self.update_from_simul(subkey,simul_data)
  
        p_datadict = {}
        idx_nan = False
        for k in self.keys:
            arr = simul_data[k].astype(JNP_DTYPE)
            arr = arr[..., slice(-policy_config["t_sample"], -1, policy_config["t_skip"])]
            if len(arr.shape) == 3:
                arr = jnp.swapaxes(arr, 1, 2)
                arr = jnp.reshape(arr, (-1, self.mparam.n_agt))
                if k != "ishock":
                    idx_nan = jnp.logical_or(idx_nan, jnp.isnan(arr).any(axis=1))
            else:
                arr = jnp.reshape(arr, (-1, 1))
                if k != "ashock":
                    idx_nan = jnp.logical_or(idx_nan, jnp.isnan(arr[:, 0]))
            p_datadict[k] = arr
        for k in self.keys:
            p_datadict[k] = p_datadict[k][~idx_nan]

        if policy_config["opt_type"] == "game":
            subkey,self.key = jax.random.split(self.key)
            p_datadict = crazyshuffle(subkey,p_datadict)

        subkey, self.key = jax.random.split(self.key)
        policy_ds = BasicDataSet(subkey, p_datadict)

        return policy_ds

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        raise NotImplementedError

class KSInitDataSet(InitDataSet):
    def __init__(self, key, mparam, config):
        subkey, key = jax.random.split(key)
        super().__init__(subkey, mparam, config)
        self.key = key
        mats = sio.loadmat(mparam.mats_path)
        self.splines = KS.construct_bspl(mats)
        self.keys = ["k_cross", "ashock", "ishock"]
        self.k_policy_bchmk = lambda k_cross, ashock, ishock: KS.k_policy_bspl(k_cross, ashock, ishock, self.splines)
        # the first burn for initialization
        self.update_with_burn(self.k_policy_bchmk, "pde")

    def get_valuedataset(self, policy, policy_type, update_init=False,model=None):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict, model=model
        )
        if update_init:
            subkey, self.key = jax.random.split(self.key)
            self.update_from_simul(subkey, simul_data)

        ashock, ishock = simul_data["ashock"], simul_data["ishock"]
        k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
        k_mean = jnp.mean(k_cross, axis=1, keepdims=True)
        # k_fm = self.compute_fm(k_cross) # n_path*n_fm*T
        discount = jnp.power(self.mparam.beta, jnp.arange(t_count,dtype=JNP_DTYPE))
        util = jnp.log(csmp)

        basic_s = jnp.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1],dtype=JNP_DTYPE)
        agt_s = jnp.zeros(shape=[0, self.mparam.n_agt, 1],dtype=JNP_DTYPE)
        value = jnp.zeros(shape=[0, self.mparam.n_agt, 1],dtype=JNP_DTYPE)
        t_idx = 0
        while t_idx + t_count < value_config['T'] - 1:
            k_tmp = k_cross[:, :, t_idx][...,jnp.newaxis]
            i_tmp = ishock[:, :, t_idx][...,jnp.newaxis]
            k_mean_tmp = jnp.repeat(k_mean[:, :, t_idx][...,jnp.newaxis], self.mparam.n_agt, axis=1)
            a_tmp = jnp.repeat(ashock[:, None, t_idx][...,jnp.newaxis], self.mparam.n_agt, axis=1)
            basic_s_tmp = jnp.concatenate([k_tmp, k_mean_tmp, a_tmp, i_tmp], axis=-1, dtype=JNP_DTYPE)
            start_indices = (0,) * (util.ndim - 1) + (t_idx,)  # (0, 0, ..., t_idx)
            slice_sizes = util.shape[:-1] + (t_count,)         # (..., t_count)
            util_slice = jax.lax.dynamic_slice(util, start_indices, slice_sizes)
            v_tmp = jnp.sum(util_slice*discount, axis=-1, keepdims=True)

            basic_s = jnp.concatenate([basic_s, basic_s_tmp], axis=0, dtype=JNP_DTYPE)
            agt_s = jnp.concatenate([agt_s, k_tmp], axis=0, dtype=JNP_DTYPE)
            value = jnp.concatenate([value, v_tmp], axis=0, dtype=JNP_DTYPE)
            t_idx += t_skip

        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        v_datadict = self.process_vdatadict(v_datadict)
        return v_datadict

    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None, model=None):
        subkey, self.key = jax.random.split(self.key)
        return KS.simul_k(subkey, n_sample, T, mparam, c_policy, policy_type, state_init, shocks, model)

@partial(jax.jit)
def crazyshuffle(key, data):
    """
    JIT-optimized row-wise shuffling with full determinism.
    
    Args:
        key: JAX PRNGKey
        data: Dict with 'k_cross' and 'ishock' arrays of same shape
        shape: Explicit static shape tuple (x,y) to enable full compilation
    
    Returns:
        Shuffled data dict with same keys
    """
    # Static shape assertion (checked at trace time)
    shape = data['k_cross'].shape
    x, y = shape

    # Vectorized permutation generation
    keys = jax.random.split(key, x)
    cols = jax.vmap(lambda k: jax.random.permutation(k, y))(keys)
    
    # Efficient batched indexing
    rows = jnp.arange(x)[:, None]  # (x,1) instead of full indices matrix
    shuffled_data = {
        "k_cross": data["k_cross"][rows, cols],
        "ishock": data["ishock"][rows, cols],
        "ashock": data["ashock"]
    }
    
    return shuffled_data