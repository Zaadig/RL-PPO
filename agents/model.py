# agents/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBase(nn.Module):
    """
    A convolutional neural network model for processing visual input.
    """

    def __init__(self, input_channels, hidden_size):
        super(CNNBase, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(3136, hidden_size),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.main(x)
        return x

class PPOAgent(nn.Module):
    """
    Combined actor-critic network for PPO.
    """

    def __init__(self, input_channels, hidden_size, num_actions):
        super(PPOAgent, self).__init__()
        self.base = CNNBase(input_channels, hidden_size)

        # Actor: outputs action probabilities
        self.actor = nn.Linear(hidden_size, num_actions)

        # Critic: outputs value prediction
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        base_output = self.base(x)

        # Actor output: action probabilities
        action_probs = F.softmax(self.actor(base_output), dim=-1)

        # Critic output: value prediction
        value = self.critic(base_output)

        return action_probs, value

