import tensorflow as tf
from models import a2c, acktr, ddpg, ppo, sac, td3, trpo

from callbacks import EarlyStopCallback
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def train_models(env, vecenv):
    seeds = [1, 2, 3]

    for seed in seeds:
        algos = [{'name': 'a2c', 'model': a2c(vecenv, seed)},
                 {'name': 'acktr', 'model': acktr(vecenv, seed)},
                 {'name': 'ddpg', 'model': ddpg(env, seed)},
                 {'name': 'ppo', 'model': ppo(vecenv, seed)},
                 {'name': 'sac', 'model': sac(env, seed)},
                 {'name': 'td3', 'model': td3(env, seed)},
                 {'name': 'trpo', 'model': trpo(env, seed)}]

        for a in algos:
            cb = EarlyStopCallback(reward_threshold=5000, verbose=1)
            early_stop = EvalCallback(env, callback_on_new_best=cb, verbose=1)

            a['model'].learn(total_timesteps=int(1e10), callback=early_stop)
            a['model'].save(f'data/models/{a["name"]}_{seed}')
            tf.reset_default_graph()
