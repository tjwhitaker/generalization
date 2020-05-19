import tensorflow as tf
from models import a2c, acktr, ddpg, ppo, sac, td3, trpo

from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def train_models(env, vecenv):
    algos = [{'name': 'a2c', 'model': a2c(vecenv)},
             {'name': 'acktr', 'model': acktr(vecenv)},
             {'name': 'ddpg', 'model': ddpg(env)},
             {'name': 'ppo', 'model': ppo(vecenv)},
             {'name': 'sac', 'model': sac(env)},
             {'name': 'td3', 'model': td3(env)},
             {'name': 'trpo', 'model': trpo(env)}]

    for a in algos:
        cb = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)
        early_stop = EvalCallback(env, callback_on_new_best=cb, verbose=1)

        a['model'].learn(total_timesteps=int(1e10), callback=early_stop)
        a['model'].save(f'data/models/{a["name"]}')
        tf.reset_default_graph()
