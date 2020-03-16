from gym import spaces
import numpy as np

class DoublePole(gym.Env):
  """
    Non-Markovian Double Pole Balancing

    Observations:
      Cart Position
      Pole 1 Angle
      Pole 2 Angle

    Actions:
      Continuous Force: [-inf, inf] N

    Dynamics:
      Track Length: [-2.4, 2.4] m
      Gravity: 9.80665 m/s
      Cart Mass 1 kg 
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
    self.gravity = 9.80665
    self.cart_mass = 1
    self.pole_1_mass = 0.1
    self.pole_2_mass = 0.01
    self.pole_1_length = 1
    self.pole_2_length = 0.1
    self.track_friction = 0.0005
    self.pole_friction = 0.000002
    self.time_step = 0.01

    # Failing State
    self.max_theta = 36 * (2 * np.pi) / 360
    self.max_position = 2.4

    # Spaces
    self.action_space = spaces.Box(-np.inf, np.inf)
    self.observation_space = spaces.Box([-max_position, -max_theta, -max_theta], [max_position, max_theta, max_theta])
    
    # State Dynamics
    # [x, dx, theta1, dtheta1, theta2, dtheta2]
    # [dx, ddx, dtheta1, ddtheta1, dtheta2, ddtheta2]
    self.state = np.array([0, 0, 0, 0, 0, 0])
    self.dstate = np.array([0, 0, 0, 0, 0, 0])

  def dynamic_system(force, state, dstate):
    pole_1_cos_theta = cos(state(2))
    pole_1_sin_theta = sin(state(2))
    pole_2_cos_theta = cos(state[4])
    pole_2_sin_theta = sin(state[4])

    pole_1_effective_mass = self.pole_1_mass * (1.0 - 0.75 * (pole_1_cos_theta**2))
    pole_1_effective_force = self.pole_1_mass * self.pole_1_length * (state[3]**2) * pole_1_sin_theta + 0.75 * self.pole_1_mass * pole_1_cos_theta * ((self.pole_friction * state[3]) / (self.pole_1_mass * self.pole_1_length) + self.gravity * pole_1_sin_theta)

    pole_2_effective_mass = self.pole_2_mass * (1.0 - 0.75 * (pole_2_cos_theta ** 2))
    pole_2_effective_force = self.pole_2_mass * self.pole_2_length * (state[5]**2) * pole_2_sin_theta + 0.75 * self.pole_2_mass * pole_2_cos_theta * ((self.pole_friction * state[5]) / (self.pole_2_mass * self.pole_2_length) + self.gravity * pole_2_sin_theta)
        
    dstate[1] = (force - self.track_friction * (state[1] > 0 ? 1.0 : -1.0) + (pole_1_effective_force + pole_2_effective_force)) / (self.cart_mass + pole_1_effective_mass + pole_2_effective_mass)
    dstate[3] = -0.75 * (dstate[1] * pole_1_cos_theta + self.gravity * pole_1_sin_theta + (self.pole_friction * state[3]) / (self.pole_1_mass * self.pole_1_length)) / self.pole_1_length
    dstate[5] = -0.75 * (dstate[1] * pole_2_cos_theta + self.gravity * pole_2_sin_theta + (self.pole_friction * state[5]) / (self.pole_2_mass * self.pole_2_length)) / self.pole_2_length

  def runge_kutta_4th(force, state, dstate):
    u1, u2, u3, u4 = [0, 0, 0, 0, 0, 0]
    state_temp, dstate_temp = [0, 0, 0, 0, 0, 0]

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
      u3[i] = self.time_step * dstate_temp

    # u4
    for i in range(len(state)):
      state_temp[i] = tate[i] + u3[i]

    dstate_temp[0] = state_temp[1]
    dstate_temp[2] = state_temp[3]
    dstate_temp[4] = state_temp[5]

    self.dynamic_system(force, state_temp, dstate_temp)

    for i in range(len(state)):
      u3[i] = self.time_step * dstate_temp

    # Final Results
    for i in range(len(state)):
      self.state[i] = state[i] + (u1[i] + 2.0 * u2[i] + 2.0 * u3[i] + u4[i]) / 6.0

    dstate[0] = state[1]
    dstate[2] = state[3]
    dstate[4] = state[5]

  def step(self, action):
    # Change state
    # runge_kutta(action, self.state, self.dstate)
    # return np.array(self.state), reward, done, {}
    pass

  def reset(self):
    # State Dynamics
    # [x, dx, theta1, dtheta1, theta2, dtheta2]
    self.state = np.array([0, 0, 0, 0, 0, 0])

  def render(self, mode='human'):
    pass