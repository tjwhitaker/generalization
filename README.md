# Generalization

```
pip3 install -e .
```

# Dependencies

Stable Baselines for RL model implementations.
https://github.com/hill-a/stable-baselines

Open AI Gym for environments.

# Usage

These train and test all 7 models. Time consuming.
```
python3 src/main.py --env single --action train
python3 src/main.py --env single --action test
python3 src/main.py --env double --action train
python3 src/main.py --env double --action test
```

```
tensorboard --logdir data/runs
```

# TODO
- Train and test single models at a time.
- Implement NK Selection Policies
- Implement training callback to save model checkpoints intelligently
- Implement Non-Markov Environments

- Implement Recurrent Policies
    - Only possible with these three algos: a2c acktr ppo
    - Worth trying?

- Test Layer Normalized Policies