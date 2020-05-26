import numpy as np
import tensorflow as tf
from stable_baselines import A2C, ACKTR, DDPG, PPO2, SAC, TD3, TRPO
from experiments import prune_policy

# TODO: Refactor all of the pruning logic


def test_models(env):
    # seeds = [1, 2, 3]
    seeds = [3]

    for s in seeds:
        # Load Models
        # models = [A2C.load(f'data/models/a2c_{s}'),
        #           ACKTR.load(f'data/models/acktr_{s}'),
        #           DDPG.load(f'data/models/ddpg_{s}'),
        #           PPO2.load(f'data/models/ppo_{s}'),
        #           SAC.load(f'data/models/sac_{s}'),
        #           TD3.load(f'data/models/td3_{s}'),
        #           TRPO.load(f'data/models/trpo_{s}')]

        models = [DDPG.load(f'data/models/ddpg_{s}')]

        for m in models:
            # run_policy(m, env)
            prune_policy(m, 0.1)
            # generalization_test(m, env)


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
