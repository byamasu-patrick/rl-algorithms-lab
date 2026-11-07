import numpy as np
import torch
import torch.nn as nn

from src.agent import layer_init


class RNDModel(nn.Module):
    """Random Network Distillation (Burda et al., 2018).

    `target` is a randomly initialised network that is never trained. `predictor` is trained to
    reproduce the target's output. On observations like those seen many times the predictor is
    accurate; on novel ones it is not, so its error is a novelty signal: the intrinsic reward.
    Both read a single normalised 84x84 frame, not the policy's 4-frame stack.
    """

    def __init__(self, input_size=1, output_size=512):
        super().__init__()
        feature_output = 7 * 7 * 64

        # Prediction network: deeper than the target, so it has the capacity to fit it
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(input_size, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, output_size)),
        )

        # Target network: a fixed random embedding of the observation
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(input_size, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, output_size)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)

    @torch.no_grad()
    def intrinsic_reward(self, next_obs):
        """Half the squared prediction error, summed over the 512 features."""
        predict_feature, target_feature = self(next_obs)
        return (target_feature - predict_feature).pow(2).sum(1) / 2


class RunningMeanStd:
    """Running mean and variance over a stream of batches (parallel algorithm of Chan et al.).
    Used for the per-pixel observation statistics and for the intrinsic-return scale."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        self.update_from_moments(np.mean(x, axis=0), np.var(x, axis=0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean, self.var, self.count = new_mean, m_2 / total_count, total_count


class RewardForwardFilter:
    """Running discounted sum of intrinsic rewards, per environment. Its spread over time is what
    the intrinsic rewards are divided by, so their scale stays stable as novelty fades."""

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
