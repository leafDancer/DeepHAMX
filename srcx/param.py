
import jax.numpy as jnp

DTYPE = "float64"
if DTYPE == "float64":
    JNP_DTYPE = jnp.float64
elif DTYPE == "float32":
    JNP_DTYPE = jnp.float32
elif DTYPE == "float16":
    JNP_DTYPE = jnp.float16
else:
    raise ValueError(f"Not support for dtype {DTYPE}")

class KSParam():
    def __init__(self, n_agt, beta, mats_path):
        self.n_agt = n_agt  # number of finite agents
        self.beta = beta  # discount factor
        self.mats_path = mats_path  # matrix from Matlab policy
        self.gamma = 1.0  # utility-function parameter
        self.alpha = 0.36  # share of capital in the production function
        self.delta = 0.025  # depreciation rate
        self.delta_a = 0.01  # (1-delta_a) is the productivity level in a bad state,
        # and (1+delta_a) is the productivity level in a good state
        self.mu = 0.15  # unemployment benefits as a share of wage
        self.l_bar = 1.0 / 0.9  # time endowment normalizes labor supply to 1 in a bad state

        self.epsilon_u = 0  # idiosyncratic shock if the agent is unemployed
        self.epsilon_e = 1  # idiosyncratic shock if the agent is employed

        self.ur_b = 0.1  # unemployment rate in a bad aggregate state
        self.er_b = (1 - self.ur_b)  # employment rate in a bad aggregate state
        self.ur_g = 0.04  # unemployment rate in a good aggregate state
        self.er_g = (1 - self.ur_g)  # employment rate in a good aggregate state

        # labor tax rate in bad and good aggregate states
        self.tau_b = self.mu * self.ur_b / (self.l_bar * self.er_b)
        self.tau_g = self.mu * self.ur_g / (self.l_bar * self.er_g)

        self.k_ss = ((1 / self.beta - (1 - self.delta)) / self.alpha) ** (1 / (self.alpha - 1))

        # steady-state capital in a deterministic model with employment rate of 0.9
        # (i.e., l_bar*L=1, where L is aggregate labor in the paper)

        self.prob_trans = jnp.array(
            [
                [0.525, 0.35, 0.03125, 0.09375],
                [0.038889, 0.836111, 0.002083, 0.122917],
                [0.09375, 0.03125, 0.291667, 0.583333],
                [0.009115, 0.115885, 0.024306, 0.850694]
            ],
            dtype=JNP_DTYPE
        )

        self.prob_ag = jnp.zeros([2, 2],dtype=JNP_DTYPE)
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