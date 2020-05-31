from random import random
import copy

from callbacks import EarlyStopCallback
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines import PPO2

import tensorflow as tf
import numpy as np
from test import generalization_test

from utils import drop_outliers


def prune_policy(class_name, params, probability):
    p = copy.deepcopy(params)

    # A2C, ACKTR, PPO, TRPO
    if class_name == "A2C" or class_name == "ACKTR" or class_name == "PPO2" or class_name == "TRPO":
        for i in range(p['model/pi_fc0/w:0'].shape[0]):
            for j in range(p['model/pi_fc0/w:0'].shape[1]):
                if random() < probability:
                    p['model/pi_fc0/w:0'][i, j] = 0

        for i in range(p['model/pi_fc1/w:0'].shape[0]):
            for j in range(p['model/pi_fc1/w:0'].shape[1]):
                if random() < probability:
                    p['model/pi_fc1/w:0'][i, j] = 0

        for i in range(p['model/pi/w:0'].shape[0]):
            for j in range(p['model/pi/w:0'].shape[1]):
                if random() < probability:
                    p['model/pi/w:0'][i, j] = 0

    if class_name == "SAC" or class_name == "TD3":
        for i in range(p['model/pi/fc0/kernel:0'].shape[0]):
            for j in range(p['model/pi/fc0/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/fc0/kernel:0'][i, j] = 0

        for i in range(p['model/pi/fc1/kernel:0'].shape[0]):
            for j in range(p['model/pi/fc1/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/fc1/kernel:0'][i, j] = 0

        for i in range(p['model/pi/dense/kernel:0'].shape[0]):
            for j in range(p['model/pi/dense/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/dense/kernel:0'][i, j] = 0

    if class_name == "DDPG":
        for i in range(p['model/pi/fc0/kernel:0'].shape[0]):
            for j in range(p['model/pi/fc0/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/fc0/kernel:0'][i, j] = 0

        for i in range(p['model/pi/fc1/kernel:0'].shape[0]):
            for j in range(p['model/pi/fc1/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/fc1/kernel:0'][i, j] = 0

        for i in range(p['model/pi/pi/kernel:0'].shape[0]):
            for j in range(p['model/pi/pi/kernel:0'].shape[1]):
                if random() < probability:
                    p['model/pi/pi/kernel:0'][i, j] = 0

    return p


def ensemble_experiment(vecenv, env):
    # First we train a full 64x64 model
    model = PPO2('MlpPolicy', env, learning_rate=0.001,
                 verbose=1, tensorboard_log="./data/runs/ensemble", seed=0)
    cb = EarlyStopCallback(reward_threshold=5000, verbose=1)
    early_stop = EvalCallback(env, callback_on_new_best=cb, verbose=1)

    model.learn(total_timesteps=int(1e10), callback=early_stop)
    model.save('data/models/ensemble/ppo_full_0')
    tf.reset_default_graph()

    # Now we train an ensemble of 8 8x8 models
    seeds = range(8)

    for seed in seeds:
        policy_kwargs = dict(net_arch=[dict(vf=[8, 8], pi=[8, 8])])
        model = PPO2('MlpPolicy', env, learning_rate=0.001,
                     verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./data/runs/ensemble", seed=seed)

        cb = EarlyStopCallback(reward_threshold=5000, verbose=1)
        early_stop = EvalCallback(env, callback_on_new_best=cb, verbose=1)

        model.learn(total_timesteps=int(1e10), callback=early_stop)
        model.save(f'data/models/ensemble/ppo_{seed}')
        tf.reset_default_graph()

# TODO: Refactor to use the generalization test function for the ensemble


def ensemble_test(env):
    # Generalization test the full model
    # model = PPO2.load('data/models/ensemble/ppo_full_0')
    # generalization_test(model, env)

    # Generalization test all the ensemble models individually
    ensemble = [PPO2.load(f'data/models/ensemble/ppo_{i}') for i in range(8)]

    # for model in ensemble:
    #     generalization_test(model, env)

    # Generalization test for the full ensemble
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
                        ensemble_actions = []
                        for model in ensemble:
                            action, _ = model.predict(observation)
                            ensemble_actions.append(action)

                        action = [np.mean(drop_outliers(ensemble_actions))]
                        observation, reward, done, info = env.step(action)

                        if done:
                            success_flag = False
                            break

                    if success_flag:
                        score = score + 1

    print(f'{model.__class__.__name__} {score}')
    tf.reset_default_graph()
