import gym
import doublepole

env = gym.make('doublepole-v0')
env.reset()
for i in range(100000):
  env.render()
  env.step(10*env.action_space.sample())
env.close()