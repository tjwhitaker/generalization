import argparse
import tensorflow as tf

from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv
from stable_baselines.common.vec_env import DummyVecEnv

from models import a2c, acktr, ddpg, ppo, sac, td3, trpo

if __name__ == '__main__'

    # Choose environment
    env = SinglePoleEnv()
    # env = DoublePoleEnv()

    # Set up models
    model = a2c(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/a2c_100k")

    model = ddpg(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/ddpg_100k")

    model = ppo(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/ppo_100k")

    model = sac(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/sac_100k")

    model = td3(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/td3_100k")

    model = trpo(env)
    model.learn(total_timesteps=100000)
    model.save("data/models/trpo_100k")