import argparse
import tensorflow as tf

import gym

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize

from stable_baselines.common.env_checker import check_env

from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv
from environments.wrappers import NormalizeWrapper
from stable_baselines.common.vec_env import DummyVecEnv

from train import train_models
from test import test_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='test')
    parser.add_argument('--env', default='single')
    args = parser.parse_args()

    # Choose environment
    if args.env == 'single':
        # env = NormalizeWrapper(SinglePoleEnv())
        env = NormalizeWrapper(gym.make('SinglePole-v0'))
    elif args.env == 'double':
        env = DoublePoleEnv()

    # TODO: Make this cleaner. Some algos require vector environments while some don't.
    vecenv = make_vec_env(SinglePoleEnv, n_envs=16,
                          wrapper_class=NormalizeWrapper)

    # TODO: Allow finer control of model selection.
    # Train or Test All Models
    if args.action == 'train':
        train_models(env, vecenv)
    elif args.action == 'test':
        test_models(env)
