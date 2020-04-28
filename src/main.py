import gym
from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from models.ppo import ppo

env = DoublePoleEnv()
# env = SinglePoleEnv()


# PPO2 requires a vectorized environment to run
env = DummyVecEnv([lambda: env])

model = ppo(MlpPolicy, env, "./ppo_cartpole_tensorboard/")
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()