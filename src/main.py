import argparse
import tensorflow as tf

from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv
from stable_baselines.common.vec_env import DummyVecEnv

from train import train_models
from test import test_models

if __name__ == '__main__':
    # Choose environment
    env = SinglePoleEnv()
    # env = DoublePoleEnv()

    # Train all models
    # train_models(env)

    # Test Models
    test_models(env)
