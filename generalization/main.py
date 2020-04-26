import gym
from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# env = DoublePoleEnv()
env = SinglePoleEnv()

# env = gym.make('CartPole-v0')

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()