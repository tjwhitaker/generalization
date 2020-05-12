import gym
from gym import spaces
import numpy as np


class DoublePoleEnv(gym.Env):
    """
      Markovian Double Pole Balancing

      Observations:
        Cart Position
        Cart Velocity
        Pole 1 Angle
        Pole 1 Velocity
        Pole 2 Angle
        Pole 2 Velocity

      Actions:
        Continuous Force: [-inf, inf] N

      Dynamics:
        Track Length: [-2.4, 2.4] m
        Gravity: -9.80665 m/s
        Cart Mass: 1 kg 
        Pole 1 Mass: 0.1 kg
        Pole 2 Mass: 0.01 kg
        Pole 1 Length: 1 m
        Pole 2 Length: 0.1 m
        Cart/Track Friction: 0.0005
        Cart/Pole Friction: 0.000002
        Time Step: 0.01
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Dynamics
        self.gravity = -9.80665
        self.cart_mass = 1
        self.pole_1_mass = 0.1
        self.pole_2_mass = 0.01
        self.pole_1_length = 1
        self.pole_2_length = 0.5
        self.track_friction = 0.0005
        self.pole_friction = 0.000002
        self.time_step = 0.01

        # State Limits
        self.max_theta = 36 * (2 * np.pi) / 360
        self.max_position = 2.4

        observation_threshold = np.array([
            self.max_position,
            np.finfo(np.float32).max,
            self.max_theta,
            np.finfo(np.float32).max,
            self.max_theta,
            np.finfo(np.float32).max])

        # Spaces
        self.action_space = spaces.Box(np.array([-1]), np.array([1]))
        self.observation_space = spaces.Box(
            -observation_threshold, observation_threshold, dtype=np.float32)

        # State Dynamics
        # [x, dx, theta1, dtheta1, theta2, dtheta2]
        # [dx, ddx, dtheta1, ddtheta1, dtheta2, ddtheta2]
        self.state = np.zeros(6)
        self.dstate = np.zeros(6)

        self.viewer = None

    def dynamic_system(self, force, state, dstate):
        pole_1_cos_theta = np.cos(state[2])
        pole_1_sin_theta = np.sin(state[2])
        pole_2_cos_theta = np.cos(state[4])
        pole_2_sin_theta = np.sin(state[4])

        pole_1_effective_mass = self.pole_1_mass * \
            (1.0 - 0.75 * (pole_1_cos_theta ** 2))
        pole_1_effective_force = self.pole_1_mass * self.pole_1_length * (state[3]**2) * pole_1_sin_theta + 0.75 * self.pole_1_mass * pole_1_cos_theta * (
            (self.pole_friction * state[3]) / (self.pole_1_mass * self.pole_1_length) + self.gravity * pole_1_sin_theta)

        pole_2_effective_mass = self.pole_2_mass * \
            (1.0 - 0.75 * (pole_2_cos_theta ** 2))
        pole_2_effective_force = self.pole_2_mass * self.pole_2_length * (state[5]**2) * pole_2_sin_theta + 0.75 * self.pole_2_mass * pole_2_cos_theta * (
            (self.pole_friction * state[5]) / (self.pole_2_mass * self.pole_2_length) + self.gravity * pole_2_sin_theta)

        dstate[1] = (force - self.track_friction * (1.0 if state[1] > 0 else -1.0) + (pole_1_effective_force +
                                                                                      pole_2_effective_force)) / (self.cart_mass + pole_1_effective_mass + pole_2_effective_mass)
        dstate[3] = -0.75 * (dstate[1] * pole_1_cos_theta + self.gravity * pole_1_sin_theta + (
            self.pole_friction * state[3]) / (self.pole_1_mass * self.pole_1_length)) / self.pole_1_length
        dstate[5] = -0.75 * (dstate[1] * pole_2_cos_theta + self.gravity * pole_2_sin_theta + (
            self.pole_friction * state[5]) / (self.pole_2_mass * self.pole_2_length)) / self.pole_2_length

    def runge_kutta(self, force, state, dstate):
        u1 = np.zeros(6)
        u2 = np.zeros(6)
        u3 = np.zeros(6)
        u4 = np.zeros(6)
        state_temp = np.zeros(6)
        dstate_temp = np.zeros(6)

        # u1
        for i in range(len(state)):
            u1[i] = self.time_step * dstate[i]

        # u2
        for i in range(len(state)):
            state_temp[i] = state[i] + (u1[i] / 2.0)

        dstate_temp[0] = state_temp[1]
        dstate_temp[2] = state_temp[3]
        dstate_temp[4] = state_temp[5]

        self.dynamic_system(force, state_temp, dstate_temp)

        for i in range(len(state)):
            u2[i] = self.time_step * dstate_temp[i]

        # u3
        for i in range(len(state)):
            state_temp[i] = state[i] + (u2[i] / 2.0)

        dstate_temp[0] = state_temp[1]
        dstate_temp[2] = state_temp[3]
        dstate_temp[4] = state_temp[5]

        self.dynamic_system(force, state_temp, dstate_temp)

        for i in range(len(state)):
            u3[i] = self.time_step * dstate_temp[i]

        # u4
        for i in range(len(state)):
            state_temp[i] = state[i] + u3[i]

        dstate_temp[0] = state_temp[1]
        dstate_temp[2] = state_temp[3]
        dstate_temp[4] = state_temp[5]

        self.dynamic_system(force, state_temp, dstate_temp)

        for i in range(len(state)):
            u4[i] = self.time_step * dstate_temp[i]

        # Final Results
        for i in range(len(state)):
            state[i] = state[i] + \
                (u1[i] + (2.0 * u2[i]) + (2.0 * u3[i] + u4[i])) / 6.0

        dstate[0] = state[1]
        dstate[2] = state[3]
        dstate[4] = state[5]

    def step(self, action):
        self.dynamic_system(10*action[0], self.state, self.dstate)
        self.runge_kutta(10*action[0], self.state, self.dstate)

        done = bool(self.state[0] < -self.max_position
                    or self.state[0] > self.max_position
                    or self.state[2] < -self.max_theta
                    or self.state[2] > self.max_theta
                    or self.state[4] < -self.max_theta
                    or self.state[4] > self.max_theta)

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return self.normalize_state(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([0, 0, 4.5 * (2 * np.pi) / 360, 0, 0, 0])
        self.dstate = np.zeros(6)

        return self.normalize_state(self.state)

    def reset_to_state(self, state):
        self.state = np.array(state)
        self.dstate = np.zeros(6)

        return self.normalize_state(self.state)

    def normalize_state(self, state):
        max_theta = 36 * (2 * np.pi) / 360
        x1 = 2 * ((state[0] + 2.4)/(4.8)) - 1
        x2 = np.tanh(state[1])
        x3 = 2 * ((state[2] + max_theta)/(2 * max_theta)) - 1
        x4 = np.tanh(state[3])
        x5 = 2 * ((state[2] + max_theta)/(2 * max_theta)) - 1
        x6 = np.tanh(state[3])

        return np.array([x1, x2, x3, x4, x5, x6])

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position*2
        scale = screen_width/world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * self.pole_1_length
        pole2len = scale * self.pole_2_length
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight/4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            l, r, t, b = -polewidth/2, polewidth/2, pole2len-polewidth/2, -polewidth/2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.6, .4, .2)
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.poletrans2)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole
            self._pole2_geom = pole2

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        pole2 = self._pole2_geom

        l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        l, r, t, b = -polewidth/2, polewidth/2, pole2len-polewidth/2, -polewidth/2
        pole2.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        self.poletrans2.set_rotation(-x[4])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
