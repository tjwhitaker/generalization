import tensorflow as tf
from models import a2c, acktr, ddpg, ppo, sac, td3, trpo

from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def train_models(env, vecenv):
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=4750, verbose=1)
    early_stop = EvalCallback(
        env, callback_on_new_best=callback_on_best, verbose=1)

    # Set up models
    # model = a2c(vecenv)
    # model.learn(total_timesteps=5000000, callback=early_stop)
    # model.save("data/models/a2c")
    # tf.reset_default_graph()

    # model = acktr(vecenv)
    # model.learn(total_timesteps=5000000, callback=early_stop)
    # model.save("data/models/acktr")
    # tf.reset_default_graph()

    # model = ddpg(env)
    # model.learn(total_timesteps=1000000, callback=early_stop)
    # model.save("data/models/ddpg")
    # tf.reset_default_graph()

    # model = ppo(vecenv)
    # model.learn(total_timesteps=5000000, callback=early_stop)
    # model.save("data/models/ppo")
    # tf.reset_default_graph()

    model = sac(env)
    model.learn(total_timesteps=1000000, callback=early_stop)
    model.save("data/models/sac")
    tf.reset_default_graph()

    # model = td3(env)
    # model.learn(total_timesteps=1000000, callback=early_stop)
    # model.save("data/models/td3")
    # tf.reset_default_graph()

    # model = trpo(env)
    # model.learn(total_timesteps=1000000, callback=early_stop)
    # model.save("data/models/trpo")
    # tf.reset_default_graph()
