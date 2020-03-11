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
    
    self.reset()

  def step(self, action):
    # Runge Kutta 4th Order
    pass

  def reset(self):
    # State Dynamics
    # [x, dx, theta1, dtheta1, theta2, dtheta2]
    self.state = np.array([0, 0, 0, 0, 0, 0])

  def render(self, mode='human'):
    pass