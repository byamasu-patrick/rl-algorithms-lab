"""RL agent implementation for reinforcement learning tasks."""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """Reinforcement learning agent with actor-critic architecture."""
    def __init__(self, envs):
        super().__init__()
        if isinstance(envs.single_action_space, gym.spaces.Box):
            action_dim = np.prod(envs.single_action_space.shape)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            action_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if hasattr(self, "actor_logstd"):
            action_logstd = self.actor_logstd.expand_as(logits)
            action_std = torch.exp(action_logstd)
            probs = Normal(logits, action_std)
        else:
            probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        if hasattr(self, "actor_logstd"):
            logprob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        else:
            logprob = probs.log_prob(action.long())
            entropy = probs.entropy()

        return action, logprob, entropy, self.critic(x)
