from stable_baselines import PPO2

def ppo(policy, env, log_dir):
  return PPO2(policy, env, verbose=1, tensorboard_log=log_dir)