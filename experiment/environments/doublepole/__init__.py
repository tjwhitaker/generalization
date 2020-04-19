from gym.envs.registration import register

register(
    id='doublepole-v0',
    entry_point='doublepole.envs:DoublePoleEnv',
)