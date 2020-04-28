import gym
from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from models.a2c import a2c
from models.acktr import acktr
from models.ddpg import ddpg
from models.gail import gail
from models.ppo import ppo
from models.sac import sac
from models.td3 import td3
from models.trpo import trpo

model = ppo(MlpPolicy, env, "./ppo_cartpole_tensorboard/")
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # (train, test)
    parser.add_argument('--action', default='test')

    # (a2c, acktr, ddpg, gail, ppo, sac, td3, trpo)
    parser.add_argument('--model', default='')

    # (single, double)
    parser.add_argument('--env', default='single')


    args = parser.parse_args()

    # Choose environment
    if args.env == 'single':
        env = SinglePoleEnv()
    elif args.env == 'double':
        env = DoublePoleEnv()

    # Set up vectorized environment
    env = DummyVecEnv([lambda: env])


