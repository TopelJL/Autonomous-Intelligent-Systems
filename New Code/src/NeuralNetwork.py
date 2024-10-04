# ---------------------------------------------------------------------
#                          NeuralNetwork.py
#                    University of Central Florida
#              Autonomous & Intelligent Systems Labratory
#
# Description: This approach to a neural network uses a Soft 
# Actor-Critic approach. This algorithm was chose for many reasons
# such as success seen in papers sighted in README.md, which our
# approach is based off of as well as SAC is well suited for
# continuous action spaces.
#
# Algorithm: Maximum Entropy Reinforcement Learning.
# Architecture: Actor-Critic (policy and value function).
# Expoloration: Entropy Regularization
# Stability: Off-policy learning and target networks.
#
#
# Author        Date        Description
# ------        ----        ------------
# Jaxon Topel   10/3/24     Initial Architecture Creation
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor network for SAC.
class Actor(nn.Module):

    # Actor Network constructor.
    def __init__(self, state_size, action_size):
        """
        Initialize the actor network.

        Args:
            state_size (int): Size of the input state.
            action_size (int): Number of possible actions.
        """
        super(Actor, self).__init__()

        # Define hidden layers (replace with your desired architecture)
        self.fc1 = nn.Linear(state_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.mu_head = nn.Linear(128, action_size)  # Mean of action distribution
        self.log_std_head = nn.Linear(128, action_size)  # Log std of action distribution

    def forward(self, state):
        """
        Forward pass through the actor network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Action means and log stds.
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = F.clamp(log_std, min=-20, max=2)  # Clamp log std for stability
        return mu, log_std

class Critic(nn.Module):
    """
    Critic network for SAC.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the critic network.

        Args:
            state_size (int): Size of the input state.
            action_size (int): Number of possible actions.
        """
        super(Critic, self).__init__()

        # Define hidden layers (replace with your desired architecture)
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.q_out = nn.Linear(128, 1)  # Output Q-value

    def forward(self, state, action):
        """
        Forward pass through the critic network.

        Args:
            state (torch.Tensor): Input state.
            action (torch.Tensor): Input action.

        Returns:
            torch.Tensor: Q-value.
        """
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.q_out(x)
        return q