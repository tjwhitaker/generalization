import numpy as np

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO


def a2c(env, seed):
    return A2C('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=seed)


def acktr(env, seed):
    return ACKTR('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=seed)


def ddpg(env, seed):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    return DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./data/runs", seed=seed)


def ppo(env, seed):
    return PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=seed)


def sac(env, seed):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    return SAC('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./data/runs", seed=seed)


def td3(env, seed):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    return TD3('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./data/runs", seed=seed)


def trpo(env, seed):
    return TRPO('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=seed)
