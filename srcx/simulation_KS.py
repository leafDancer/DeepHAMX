import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from param import JNP_DTYPE

EPSILON = 1e-3

@partial(jax.jit, static_argnames=("mparam","n_sample","T"))
def simul_shocks(key, n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    ashock = jnp.zeros([n_sample, T], dtype=JNP_DTYPE)
    ishock = jnp.ones([n_sample, n_agt, T], dtype=JNP_DTYPE)
    def is_state_init(key, ashock, ishock, state_init, mparam):
        del key
        if state_init is None:
            return ashock, ishock 
        ashock = ashock.at[:, 0:1].set(((state_init["ashock"] - 1) / mparam.delta_a + 1) / 2)
        ishock = ishock.at[..., 0].set(state_init["ishock"])
        return ashock, ishock

    def not_state_init(key, ashock, ishock, n_sample, n_agt, mparam):
        subkey, key = jax.random.split(key)
        ashock = ashock.at[:, 0].set(jax.random.binomial(subkey, 1, 0.5, n_sample))
        ur_rate = ashock[:, 0] * mparam.ur_g + (1 - ashock[:, 0]) * mparam.ur_b
        ur_rate = jnp.repeat(ur_rate[:, None], n_agt, axis=1)
        subkey, key = jax.random.split(key)
        rand = jax.random.uniform(subkey, shape=(n_sample, n_agt), dtype=JNP_DTYPE)
        ishock0 = jnp.where(rand < ur_rate, jnp.zeros_like(ishock[..., 0], dtype=JNP_DTYPE), ishock[..., 0])
        ishock = ishock.at[..., 0].set(ishock0)
        return ashock, ishock

    subkey, key = jax.random.split(key)
    ashock, ishock = jax.lax.cond(
        state_init is not None,
        lambda k, a, i: is_state_init(k, a, i, state_init, mparam),
        lambda k, a, i: not_state_init(k, a, i, n_sample, n_agt, mparam),
        subkey,
        ashock,
        ishock
    )
    def assign_ashock(t,val):
        ashock,key = val
        subkey,key = jax.random.split(key)
        if_keep = jax.random.binomial(subkey, 1, 0.875, n_sample, dtype=JNP_DTYPE)  # prob for Z to stay the same is 0.875
        ashock = ashock.at[:, t].set(if_keep * ashock[:, t - 1] + (1 - if_keep) * (1 - ashock[:, t - 1]))
        return ashock,key
    ashock,key = jax.lax.fori_loop(1,T,assign_ashock,(ashock,key))
    def assign_ishock(t,val):
        ishock,ashock,key = val
        a0, a1 = ashock[:, None, t - 1], ashock[:, None, t]
        y_agt = ishock[:, :, t - 1]
        ur_rate = (1 - a0) * (1 - a1) * (1 - y_agt) * mparam.p_bb_uu + (1 - a0) * (1 - a1) * y_agt * mparam.p_bb_eu
        ur_rate += (1 - a0) * a1 * (1 - y_agt) * mparam.p_bg_uu + (1 - a0) * a1 * y_agt * mparam.p_bg_eu
        ur_rate += a0 * (1 - a1) * (1 - y_agt) * mparam.p_gb_uu + a0 * (1 - a1) * y_agt * mparam.p_gb_eu
        ur_rate += a0 * a1 * (1 - y_agt) * mparam.p_gg_uu + a0 * a1 * y_agt * mparam.p_gg_eu
        subkey,key = jax.random.split(key)
        rand = jax.random.uniform(subkey,shape=(n_sample, n_agt), dtype=JNP_DTYPE)
        ishockt = jnp.where(rand<ur_rate,jnp.zeros_like(ishock[...,t]),ishock[...,t])
        ishock = ishock.at[..., t].set(ishockt)
        return ishock,ashock,key
    
    ishock,ashock,key = jax.lax.fori_loop(1,T,assign_ishock,(ishock,ashock,key))
    ashock = (ashock * 2 - 1) * mparam.delta_a + 1  # convert 0/1 variable to productivity
    return ashock, ishock

def simul_k(key, n_sample, T, mparam, policy, policy_type, state_init=None, shocks=None, model=None):
    # policy_type: "pde" or "nn_share"
    # return k_cross [n_sample, n_agt, T]
    assert policy_type in ["pde", "nn_share"], "Invalid policy type"
    n_agt = mparam.n_agt
    if shocks:
        ashock, ishock = shocks
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert jnp.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                jnp.array_equal(ishock[..., 0], state_init["ishock"]), \
                "Shock inputs are inconsistent with state_init"
    else:
        subkey,key = jax.random.split(key)
        ashock, ishock = simul_shocks(subkey, n_sample, T, mparam, state_init)

    k_cross = jnp.zeros([n_sample, n_agt, T], dtype=JNP_DTYPE)
    if state_init is not None:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross = k_cross.at[:, :, 0].set( state_init["k_cross"] )
    else:
        k_cross = k_cross.at[:, :, 0].set( mparam.k_ss )
    csmp = jnp.zeros([n_sample, n_agt, T-1], dtype=JNP_DTYPE)
    wealth = k_cross.copy()
    if policy_type == "pde":
        def loop_pde(t,pack):
            wealth,k_cross,ashock,ishock,csmp = pack
            wealth = wealth.at[:, :, t].set(next_wealth(k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1], mparam))
            k_cross_t = policy(k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1])
            # avoid csmp being too small or even negative
            k_cross = k_cross.at[:, :, t].set( jnp.clip(k_cross_t, EPSILON, wealth[:, :, t]-jnp.minimum(1.0, 0.8*wealth[:, :, t])) )
            csmp = csmp.at[:, :, t-1].set( wealth[:, :, t] - k_cross[:, :, t] )
            return wealth,k_cross,ashock,ishock,csmp
        wealth,k_cross,ashock,ishock,csmp = jax.lax.fori_loop(1,T,loop_pde,(wealth,k_cross,ashock,ishock,csmp))

    if policy_type == "nn_share" and model != None:
        def loop_nn(t,pack):
            wealth,k_cross,ashock,ishock,csmp = pack
            wealth = wealth.at[:, :, t].set(next_wealth(k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1], mparam))
            csmp_t = policy(model[0],model[1],k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1]) * wealth[:, :, t]
            #csmp_t = policy(model[0],model[1],k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1]) * wealth[:, :, t]
            csmp_t = jnp.clip(csmp_t, EPSILON, wealth[:, :, t]-EPSILON)
            k_cross = k_cross.at[:, :, t].set(wealth[:, :, t] - csmp_t)
            csmp = csmp.at[:, :, t-1].set(csmp_t)
            return wealth,k_cross,ashock,ishock,csmp
        wealth,k_cross,ashock,ishock,csmp = jax.lax.fori_loop(1,T,loop_nn,(wealth,k_cross,ashock,ishock,csmp))
    if policy_type == "nn_share" and model == None:
        def loop_nn(t,pack):
            wealth,k_cross,ashock,ishock,csmp = pack
            wealth = wealth.at[:, :, t].set(next_wealth(k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1], mparam))
            csmp_t = policy(k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1]) * wealth[:, :, t]
            #csmp_t = policy(model[0],model[1],k_cross[:, :, t-1], ashock[:, t-1].reshape(-1,1), ishock[:, :, t-1]) * wealth[:, :, t]
            csmp_t = jnp.clip(csmp_t, EPSILON, wealth[:, :, t]-EPSILON)
            k_cross = k_cross.at[:, :, t].set(wealth[:, :, t] - csmp_t)
            csmp = csmp.at[:, :, t-1].set(csmp_t)
            return wealth,k_cross,ashock,ishock,csmp
        wealth,k_cross,ashock,ishock,csmp = jax.lax.fori_loop(1,T,loop_nn,(wealth,k_cross,ashock,ishock,csmp))
    simul_data = {"k_cross": k_cross, "csmp": csmp, "ashock": ashock, "ishock": ishock}
    return simul_data

@partial(jax.jit, static_argnames=("mparam",))
def next_wealth(k_cross, ashock, ishock, mparam):
    k_mean = jnp.mean(k_cross, axis=1, keepdims=True)
    tau = jnp.where(ashock < 1, mparam.tau_b, mparam.tau_g)  # labor tax rate based on ashock
    emp = jnp.where(ashock < 1, mparam.l_bar*mparam.er_b, mparam.l_bar*mparam.er_g)  # total labor supply based on ashock
    R = 1 - mparam.delta + ashock * mparam.alpha*(k_mean / emp)**(mparam.alpha-1)
    wage = ashock*(1-mparam.alpha)*(k_mean / emp)**(mparam.alpha)
    wealth = R * k_cross + (1-tau)*wage*mparam.l_bar*ishock + mparam.mu*wage*(1-ishock)
    return wealth

def k_policy_bspl(k_cross, ashock, ishock, splines):
    k_next = jnp.zeros_like(k_cross, dtype=JNP_DTYPE)
    k_mean = jnp.repeat(jnp.mean(k_cross, axis=1, keepdims=True), k_cross.shape[1], axis=1)

    def interp_func(spline, xq, yq):
        points = jnp.stack([xq, yq], axis=-1, dtype=JNP_DTYPE)
        return spline(points)

    # parallely handle all cases
    def apply_case(mask, spline_key):
        k_tmp = jnp.where(mask, k_cross, 0.0)
        km_tmp = jnp.where(mask, k_mean, 0.0)
        return interp_func(splines[spline_key], k_tmp, km_tmp) * mask

    k_next += apply_case((ashock < 1) & (ishock == 0), '00')
    k_next += apply_case((ashock < 1) & (ishock == 1), '01')
    k_next += apply_case((ashock > 1) & (ishock == 0), '10')
    k_next += apply_case((ashock > 1) & (ishock == 1), '11')

    return k_next

def construct_bspl(mats):
    def psedo_RBS(x,y,z):
        '''jax.scipy.interpolate only support linear interpolation.'''
        return jscipy.interpolate.RegularGridInterpolator((x,y),z,method="linear")
    # mats is saved in Matlab through
    # "save(filename, 'kprime', 'k', 'km', 'agshock', 'idshock', 'kmts', 'kcross');"
    splines = {
        '00': psedo_RBS(mats['k'][...,0], mats['km'][...,0], mats['kprime'][:, :, 0, 0]),
        '01': psedo_RBS(mats['k'][...,0], mats['km'][...,0], mats['kprime'][:, :, 0, 1]),
        '10': psedo_RBS(mats['k'][...,0], mats['km'][...,0], mats['kprime'][:, :, 1, 0]),
        '11': psedo_RBS(mats['k'][...,0], mats['km'][...,0], mats['kprime'][:, :, 1, 1]),
    }
    return splines

