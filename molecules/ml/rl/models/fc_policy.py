import torch
import torch.nn as nn
import torch.nn.functional as F


class FCPolicy(nn.Module):
    """
    Simple fully connected policy network.
    """
    def __init__(self, state_dim, n_actions):
        super(FCPolicy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 128)
        self.affine2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
