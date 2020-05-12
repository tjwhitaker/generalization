from models import a2c, acktr, ddpg, ppo, sac, td3, trpo


def train_models(env):
    # Set up models
    model = a2c(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/a2c_10k")

    model = acktr(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/acktr_10k")

    model = ddpg(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/ddpg_10k")

    model = ppo(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/ppo_10k")

    model = sac(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/sac_10k")

    model = td3(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/td3_10k")

    model = trpo(env)
    model.learn(total_timesteps=10000)
    model.save("data/models/trpo_10k")
