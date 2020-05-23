import gym
from .doublepole_env import DoublePoleEnv
from gym.envs.registration import register


# Prevent Errors with Re-Registrations on Import
def register_env(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=5000,
    )


register_env(
    id='DoublePole-v0',
    entry_point='src.environments.doublepole:DoublePoleEnv'
)
