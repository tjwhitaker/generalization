from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgmlp
from stable_baselines.sac.policies import MlpPolicy as sacmlp
from stable_baselines.td3.policies import MlpPolicy as td3mlp

from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO

def a2c(env):
  return A2C(MlpPolicy, env, verbose=1, tensorboard_log="./data/runs")

def acktr(env):
  return ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="./data/runs")

def ddpg(env):
  return DDPG(ddpgmlp, env, verbore=1, tensorboard_log="./data/runs")

def ppo(env):
  return PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./data/runs")

def sac(env):
  return SAC(sacmlp, env, verbose=1, tensorboard_log="./data/runs")

def td3(env):
  return TD3(td3mlp, env, verbose=1, tensorboard_log="./data/runs")

def trpo(env):
  return TRPO(MlpPolicy, env, verbose=1, tensorboard_log="./data/runs")