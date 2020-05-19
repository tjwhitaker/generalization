import numpy as np

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO


# Tuned
def a2c(env):
    return A2C('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)


# Tuned
def acktr(env):
    return ACKTR('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)


# Not Learning?
def ddpg(env):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    return DDPG('MlpPolicy', env, action_noise=action_noise, n_cpu_tf_sess=None, verbose=1, tensorboard_log="./data/runs", seed=0)


# Tuned
def ppo(env):
    return PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)


# ???
def sac(env):
    return SAC('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)


# ???
def td3(env):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    return TD3('MlpPolicy', env, action_noise=action_noise, tensorboard_log="./data/runs", seed=0)


# ???
def trpo(env):
    return TRPO('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)
