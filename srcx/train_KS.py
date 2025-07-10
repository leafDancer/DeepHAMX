import jax
print(jax.local_devices())
from absl import flags, app
from param import KSParam,DTYPE
import json
import os 
import time 
from dataset import KSInitDataSet
from util import print_elapsedtime
from value import ValueTrainer
from policy import KSPolicyTrainer

flags.DEFINE_string("config_path", "./configs/KS/game_nn_n50.json",
                    """The path to load json file.""",
                    short_name='c')
flags.DEFINE_string("exp_name", "test",
                    """The suffix used in model_path for save.""",
                    short_name='n')
FLAGS = flags.FLAGS
if DTYPE == "float64":
    jax.config.update("jax_enable_x64", True)

def main(argv):
    start_time = time.monotonic()
    del argv
    with open(FLAGS.config_path, 'r') as f:
        config = json.load(f)
    print("Solving the problem based on the config path {}".format(FLAGS.config_path))
    mparam = KSParam(config["n_agt"], config["beta"], config["mats_path"])
    # save config at the beginning for checking
    model_path = "../data/simul_results/KS/{}_{}_n{}_{}".format(
        "game" if config["policy_config"]["opt_type"] == "game" else "sp",
        config["dataset_config"]["value_sampling"],
        config["n_agt"],
        FLAGS.exp_name,
    )
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "config_beg.json"), 'w') as f:
        json.dump(config, f)

    # initial value training
    key = jax.random.PRNGKey(41)
    subkey,key = jax.random.split(key)
    init_ds = KSInitDataSet(subkey, mparam, config)
    value_config = config["value_config"]
    if config["init_with_bchmk"]:
        init_policy = init_ds.k_policy_bchmk
        policy_type = "pde"
        # TODO: change all "pde" to "conventional"
    else:
        init_policy = init_ds.c_policy_const_share
        policy_type = "nn_share"
    vds = init_ds.get_valuedataset(
        init_policy, policy_type, 
        update_init=False,
    )
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    vds = vtrainers[0].prepare_state_v(vds)
    for vtr in vtrainers:
        subkey,key = jax.random.split(key)
        vtr.train(subkey, value_config["num_epoch"],value_config["batch_size"], vds)

    # iterative policy and value training
    policy_config = config["policy_config"]
    subkey,key = jax.random.split(key)
    ptrainer = KSPolicyTrainer(subkey,vtrainers, init_ds)
    subkey,key = jax.random.split(key)
    ptrainer.train(subkey, policy_config["num_step"], policy_config["batch_size"])

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)

if __name__ == '__main__':
    app.run(main)