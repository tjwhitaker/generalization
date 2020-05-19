import numpy as np

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO
from stable_baselines.ddpg.policies import MlpPolicy


# Tuned
def a2c(env):
    return A2C('MlpPolicy', env, lr_schedule="linear", gamma=0.999, ent_coef=0.001, learning_rate=0.001, verbose=1, tensorboard_log="./data/runs", seed=0)


# Tuned
def acktr(env):
    return ACKTR('MlpPolicy', env, gamma=0.999, ent_coef=0.001, verbose=1, tensorboard_log="./data/runs", seed=0)


# Not Learning?
def ddpg(env):
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    return DDPG(MlpPolicy, env, buffer_size=1000000, action_noise=action_noise, n_cpu_tf_sess=None, verbose=1, tensorboard_log="./data/runs", seed=0)


# Tuned
def ppo(env):
    return PPO2('MlpPolicy', env, gamma=0.999, ent_coef=0.001, noptepochs=16, verbose=1, tensorboard_log="./data/runs", seed=0)


# ???
def sac(env):
    return SAC('MlpPolicy', env, verbose=1, tensorboard_log="./data/runs", seed=0)


# ???
def td3(env):
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1), sigma=0.5 * np.ones(1))

    return TD3('MlpPolicy', env, gamma=0.999, verbose=1, action_noise=action_noise, tensorboard_log="./data/runs", seed=0)


# ???
def trpo(env):
    return TRPO('MlpPolicy', env, gamma=0.999, verbose=1, tensorboard_log="./data/runs", seed=0)
