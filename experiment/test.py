from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg_tf1, trpo_tf1, ppo_tf1, ddpg_tf1, td3_tf1, sac_tf1
import tensorflow as tf
import gym

def env_fn():
    import doublepole
    return gym.make('doublepole-v0')

def train(args):
    algos = [vpg_tf1, trpo_tf1, ppo_tf1, ddpg_tf1, td3_tf1, sac_tf1]

    for algo in algos:
        eg = ExperimentGrid(name=algo.__name__)
        eg.add('env_fn', env_fn)
        eg.add('epochs', args.epochs)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('ac_kwargs:hidden_sizes', (16,16), 'hid')
        eg.add('ac_kwargs:activation', tf.nn.relu)
        eg.run(algo, num_cpu=args.cpu)

# Generalization Test
def test():
    # Load Model
    # States
    # cart position (-2.14, 2.14)m
    # cart velocity (-1.35, 1.35)m
    # pole 1 pos (-8.6, 8.6) deg
    # pole 1 vel (-3.6, 3.6) deg/s
    # pole 2 always 0 0 angle and vel

    ranges = [4.28, 2.70, 17.2, 7.2]
    percentages = [0.05, 0.25, 0.50, 0.75, 0.95]

    env = env_fn()

    score = 0

    for s0 in percentages:
        for s1 in percentages:
            for s2 in percentages:
                for s3 in percentages:
                    state = [0, 0, 0, 0, 0, 0]
                    state[0] = (s0 * ranges[0] - ranges[0] / 2)
                    state[1] = (s1 * ranges[1] - ranges[1] / 2)
                    state[2] = (s2 * ranges[2] - ranges[2] / 2) * 2 * np.pi / 360
                    state[3] = (s3 * ranges[3] - ranges[3] / 2) * 2 * np.pi / 360

                    # Run test(policy)
                    success_flag = True
                    observation = env.reset_to_state(state)
                    for i in range(1000):
                        env.render()
                        #action = model output (observation)
                        observation, reward, done, info = env.step(action)
                        
                        if done:
                            success_flag = False
                            break
                    
                    if success_flag:
                        score = score + 1
    
    print(score)
    return score
                        




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    train(args)
    test()

    