from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_tf1, ddpg_tf1
import tensorflow as tf
import gym

def env_fn():
    import doublepole
    return gym.make('doublepole-v0')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ddpg-tf1-bench')
    eg.add('env_fn', env_fn)
    eg.add('epochs', 1)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('ac_kwargs:hidden_sizes', (64,64), 'hid')
    eg.add('ac_kwargs:activation', tf.nn.relu)
    eg.run(ddpg_tf1, num_cpu=args.cpu)