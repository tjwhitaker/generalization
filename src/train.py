import tensorflow as tf
from models import a2c, acktr, ddpg, ppo, sac, td3, trpo


def train_models(env):
    # Set up models
    model = a2c(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/a2c_500k")
    tf.reset_default_graph()

    model = acktr(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/acktr_500k")
    tf.reset_default_graph()

    model = ddpg(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/ddpg_500k")
    tf.reset_default_graph()

    model = ppo(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/ppo_500k")
    tf.reset_default_graph()

    model = sac(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/sac_500k")
    tf.reset_default_graph()

    model = td3(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/td3_500k")
    tf.reset_default_graph()

    model = trpo(env)
    model.learn(total_timesteps=500000)
    model.save("data/models/trpo_500k")
    tf.reset_default_graph()
