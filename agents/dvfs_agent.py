from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from agents.environment import DVFSEnvironment
import torch.nn as nn
import numpy as np


class EarlyStopCallback(BaseCallback):
    def __init__(self, max_no_improvement_evals=5, min_evals=20, verbose=0):
        super().__init__(verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        if self.n_calls < self.min_evals:
            return True

        # Get current reward
        mean_reward = np.mean(self.training_env.get_attr("rewards")[-100:])

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.max_no_improvement_evals:
            if self.verbose > 0:
                print(
                    f"Stopping training - no improvement in {self.max_no_improvement_evals} evaluations"
                )
            return False

        return True


def train_dvfs_agent(
    env: DVFSEnvironment, steps: int = 200_000, checkpoint_callback=None
):
    """Enhanced training configuration"""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.2,
        clip_range=0.3,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.ReLU,
        ),
        verbose=1,
    )

    return model.learn(
        total_timesteps=steps, progress_bar=True, callback=checkpoint_callback
    )
