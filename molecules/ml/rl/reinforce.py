import torch
from torch.distributions import Categorical


class Reinforce:
    """
    Reinforce algorithm.

    Notes:
    -----
    * This algorithm in stochastic. We sample from a
      categorical distribution of softmax scores from the
      policy network.
    """
    def __init__(self, policy):
        self.policy = policy

    def select_action(self, state):
        """
        Choose the next action.

        Parameters:
        ----------
        * `state` : ndarray
            Current state of the environment
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self, optimizer):
        """
        Append rewards and apply gradients.

        Parameters:
        ----------
        * `optimizer` torch.optimizer
        """
        R = 0
        policy_loss = []
        rewards = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for log_prob, reward in zip(policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
