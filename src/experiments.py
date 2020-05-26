from random import random


def prune_policy(model, probability):
    # model = A2C.load('data/models/a2c_1')
    # model = ACKTR.load(f'data/models/acktr_1')
    # model = DDPG.load(f'data/models/ddpg_1')  # Different params
    # model = PPO2.load(f'data/models/ppo_1')
    # model = SAC.load(f'data/models/sac_1')
    # model = TD3.load(f'data/models/td3_1')
    # model = TRPO.load(f'data/models/trpo_1')

    params = model.get_parameters()
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

    # A2C, ACKTR, PPO, TRPO
    # for i in range(params['model/pi_fc0/w:0'].shape[0]):
    #     for j in range(params['model/pi_fc0/w:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi_fc0/w:0'][i, j] = 0

    # for i in range(params['model/pi_fc1/w:0'].shape[0]):
    #     for j in range(params['model/pi_fc1/w:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi_fc1/w:0'][i, j] = 0

    # for i in range(params['model/pi/w:0'].shape[0]):
    #     for j in range(params['model/pi/w:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi/w:0'][i, j] = 0

    # SAC, TD3
    # for i in range(params['model/pi/fc0/kernel:0'].shape[0]):
    #     for j in range(params['model/pi/fc0/kernel:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi/fc0/kernel:0'][i, j] = 0

    # for i in range(params['model/pi/fc1/kernel:0'].shape[0]):
    #     for j in range(params['model/pi/fc1/kernel:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi/fc1/kernel:0'][i, j] = 0

    # for i in range(params['model/pi/dense/kernel:0'].shape[0]):
    #     for j in range(params['model/pi/dense/kernel:0'].shape[1]):
    #         if random() < probability:
    #             params['model/pi/dense/kernel:0'][i, j] = 0

    # DDPG
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
