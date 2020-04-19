from gym.envs.registration import register

register(
    id='singlepole-v0',
    entry_point='singlepole.envs:SinglePoleEnv',
)