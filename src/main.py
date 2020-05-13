import argparse
import tensorflow as tf

from stable_baselines.common import make_vec_env

from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv
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
        env = SinglePoleEnv()
    elif args.env == 'double':
        env = DoublePoleEnv()

    # Train or Test All Models
    if args.action == 'train':
        train_models(env)
    elif args.action == 'test':
        test_models(env)
