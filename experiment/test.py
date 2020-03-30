from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch
import gym

def env_fn():
    import doublepole  # registers custom envs to gym env registry
    return gym.make('doublepole-v0')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_fn', env_fn)
    eg.add('epochs', 500)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', (32,), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.ReLU)
    eg.run(ppo_pytorch, num_cpu=args.cpu)