from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.test_policy import load_policy_and_env
from spinup import vpg_tf1, trpo_tf1, ppo_tf1, ddpg_tf1, td3_tf1, sac_tf1
import numpy as np
import tensorflow as tf
import gym

def env_fn():
    import doublepole
    return gym.make('doublepole-v0')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    algos = [vpg_tf1, trpo_tf1, ppo_tf1, ddpg_tf1, td3_tf1, sac_tf1]

    for algo in algos:
        eg = ExperimentGrid(name=algo.__name__)
        eg.add('env_fn', env_fn)
        eg.add('epochs', args.epochs)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('ac_kwargs:hidden_sizes', (16,16), 'hid')
        eg.add('ac_kwargs:activation', tf.nn.tanh)
        eg.run(algo, num_cpu=args.cpu)