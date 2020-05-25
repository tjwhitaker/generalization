import numpy as np
import tensorflow as tf
from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO


def test_models(env):
    seeds = [1, 2, 3]

    for s in seeds:
        # Load Models
        models = [A2C.load(f'data/models/a2c_{s}'),
                  ACKTR.load(f'data/models/acktr_{s}'),
                  DDPG.load(f'data/models/ddpg_{s}'),
                  PPO2.load(f'data/models/ppo_{s}'),
                  SAC.load(f'data/models/sac_{s}'),
                  TD3.load(f'data/models/td3_{s}'),
                  TRPO.load(f'data/models/trpo_{s}')]

        for m in models:
            # run_policy(m, env)
            generalization_test(m, env)


def prune_policy():
    # model = A2C.load('data/models/a2c_1')
    # model = ACKTR.load(f'data/models/acktr_1')
    # model = DDPG.load(f'data/models/ddpg_1')  # Different params
    # model = PPO2.load(f'data/models/ppo_1')
    # model = SAC.load(f'data/models/sac_1')
    # model = TD3.load(f'data/models/td3_1')
    # model = TRPO.load(f'data/models/trpo_1')

    # params = model.get_parameters()
    # print(params.keys())

    # Policy network parameters: A2C, ACKTR, PPO, TRPO
    # print(np.shape(params['model/pi_fc0/w:0']))  # 4x64
    # print(np.shape(params['model/pi_fc1/w:0']))  # 64x64
    # print(np.shape(params['model/pi/w:0']))      # 64x1

    # Policy network parameters: DDPG
    # print(np.shape(params['model/pi/fc0/kernel:0']))  # 4x64
    # print(np.shape(params['model/pi/fc1/kernel:0']))  # 64x64
    # print(np.shape(params['model/pi/pi/kernel:0']))   # 64x1

    # Policy network parameters: SAC, TD3
    # print(np.shape(params['model/pi/fc0/kernel:0']))  # 4x64
    # print(np.shape(params['model/pi/fc1/kernel:0']))  # 64x64
    # print(np.shape(params['model/pi/dense/kernel:0']))   # 64x1


def run_policy(model, env):
    observation = env.reset()

    for i in range(1000):
        env.render()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)

        if done:
            break


# Set up for single pole env
# TODO: Parametrize to make usable for single/double markov/nonmarkov

def generalization_test(model, env):
    ranges = [4.28, 2.70, 17.2, 7.2]
    percentages = [0.05, 0.25, 0.50, 0.75, 0.95]
    score = 0

    for s0 in percentages:
        for s1 in percentages:
            for s2 in percentages:
                for s3 in percentages:
                    state = [0, 0, 0, 0]
                    state[0] = (s0 * ranges[0] - ranges[0] / 2)
                    state[1] = (s1 * ranges[1] - ranges[1] / 2)
                    state[2] = (s2 * ranges[2] - ranges[2] /
                                2) * 2 * np.pi / 360
                    state[3] = (s3 * ranges[3] - ranges[3] /
                                2) * 2 * np.pi / 360

                    # Run test(policy)
                    success_flag = True

                    env.reset()
                    observation = env.reset_to_state(state)

                    for i in range(1000):
                        action, _states = model.predict(observation)
                        observation, reward, done, info = env.step(action)

                        if done:
                            success_flag = False
                            break

                    if success_flag:
                        score = score + 1

    print(score)
    tf.reset_default_graph()
