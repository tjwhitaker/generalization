import argparse

from environments.singlepole import SinglePoleEnv
from environments.doublepole import DoublePoleEnv
from stable_baselines.common.vec_env import DummyVecEnv

from models import a2c, acktr, ddpg, ppo, sac, td3, trpo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # (train, test)
    parser.add_argument('--action', default='test')

    # (a2c, acktr, ddpg, ppo, sac, td3, trpo)
    parser.add_argument('--model', default='')

    # (mlp, lstm, lnlstm)
    parser.add_argument('-policy', default='mlp')

    # (single, double)
    parser.add_argument('--env', default='single')

    args = parser.parse_args()

    # Choose environment
    if args.env == 'single':
        env = SinglePoleEnv()
    elif args.env == 'double':
        env = DoublePoleEnv()

    # Set up vectorized environment
    env = DummyVecEnv([lambda: env])

    # Set up model
    model = ppo(env)

    model.learn(total_timesteps=50000)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()