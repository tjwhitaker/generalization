from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.test_policy import load_policy_and_env
from spinup import vpg_tf1, trpo_tf1, ppo_tf1, ddpg_tf1, td3_tf1, sac_tf1
import numpy as np
import tensorflow as tf
import gym

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
    models = ["/home/tim/Research/generalization/spinningup/data/vpg/vpg_s0/",
              "/home/tim/Research/generalization/spinningup/data/trpo/trpo_s0/",
              "/home/tim/Research/generalization/spinningup/data/ppo/ppo_s0/",
              "/home/tim/Research/generalization/spinningup/data/ddpg/ddpg_s0/",
              "/home/tim/Research/generalization/spinningup/data/td3/td3_s0/",
              "/home/tim/Research/generalization/spinningup/data/sac/sac_s0/"]

    for model in models:
        score = 0

        env, get_action = load_policy_and_env(model, "last", False)

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
                            # env.render()
                            action = get_action(observation)
                            observation, reward, done, info = env.step(action)
                            
                            if done:
                                success_flag = False
                                break
                        
                        if success_flag:
                            score = score + 1
    
        print(model, score)
        tf.reset_default_graph()

if __name__ == '__main__':
    test()