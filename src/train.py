def train(model, env, steps):
  model.learn(total_timesteps=steps)