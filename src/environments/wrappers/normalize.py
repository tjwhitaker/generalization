import numpy as np
import gym


class NormalizeWrapper(gym.Wrapper):
    """
    Normalizes the observation space of the single pole environment.
    """

    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.normalize(state), reward, done, info

    def reset(self):
        state = self.env.reset()
        return self.normalize(state)

    def reset_to_state(self, state):
        state = self.env.reset_to_state(state)
        return self.normalize(state)

    def normalize(self, state):
        max_theta = 36 * (2 * np.pi) / 360
        x1 = 2 * ((state[0] + 2.4)/(4.8)) - 1
        x2 = 2 * ((state[1] + 3)/(2 * 3)) - 1
        x3 = 2 * ((state[2] + max_theta)/(2 * max_theta)) - 1
        x4 = 2 * ((state[3] + 3)/(2 * 3)) - 1

        # Double Pole
        # x5 = 2 * ((state[2] + max_theta)/(2 * max_theta)) - 1
        # x6 = 2 * ((state[3] + 3)/(2 * 3)) - 1

        return np.array([x1, x2, x3, x4])
