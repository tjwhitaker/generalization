from stable_baselines.common.callbacks import BaseCallback


class EarlyStopCallback(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).
    It must be used with the `EvalCallback`.
    :param reward_threshold: (float)  Minimum expected reward per episode
        to stop training.
    :param verbose: (int)
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super(EarlyStopCallback, self).__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, ("`EarlyStop` callback must be used "
                                         "with an `EvalCallback`")
        # Convert np.bool to bool, otherwise callback.on_step() is False won't work
        continue_training = bool(
            self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print("Stopping training because the mean reward {:.2f} "
                  " is above the threshold {}".format(self.parent.best_mean_reward, self.reward_threshold))
        return continue_training
