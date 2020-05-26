from random import random


def prune_policy(model, probability):
    class_name = model.__class__.__name__
    params = model.get_parameters()

    # A2C, ACKTR, PPO, TRPO
    if class_name == "A2C" or class_name == "ACKTR" or class_name == "PPO2" or class_name == "TRPO":
        for i in range(params['model/pi_fc0/w:0'].shape[0]):
            for j in range(params['model/pi_fc0/w:0'].shape[1]):
                if random() < probability:
                    params['model/pi_fc0/w:0'][i, j] = 0

        for i in range(params['model/pi_fc1/w:0'].shape[0]):
            for j in range(params['model/pi_fc1/w:0'].shape[1]):
                if random() < probability:
                    params['model/pi_fc1/w:0'][i, j] = 0

        for i in range(params['model/pi/w:0'].shape[0]):
            for j in range(params['model/pi/w:0'].shape[1]):
                if random() < probability:
                    params['model/pi/w:0'][i, j] = 0

    if class_name == "SAC" or class_name == "TD3":
        for i in range(params['model/pi/fc0/kernel:0'].shape[0]):
            for j in range(params['model/pi/fc0/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/fc0/kernel:0'][i, j] = 0

        for i in range(params['model/pi/fc1/kernel:0'].shape[0]):
            for j in range(params['model/pi/fc1/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/fc1/kernel:0'][i, j] = 0

        for i in range(params['model/pi/dense/kernel:0'].shape[0]):
            for j in range(params['model/pi/dense/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/dense/kernel:0'][i, j] = 0

    if class_name == "DDPG":
        for i in range(params['model/pi/fc0/kernel:0'].shape[0]):
            for j in range(params['model/pi/fc0/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/fc0/kernel:0'][i, j] = 0

        for i in range(params['model/pi/fc1/kernel:0'].shape[0]):
            for j in range(params['model/pi/fc1/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/fc1/kernel:0'][i, j] = 0

        for i in range(params['model/pi/pi/kernel:0'].shape[0]):
            for j in range(params['model/pi/pi/kernel:0'].shape[1]):
                if random() < probability:
                    params['model/pi/pi/kernel:0'][i, j] = 0

    model.load_parameters(params)

    return model
