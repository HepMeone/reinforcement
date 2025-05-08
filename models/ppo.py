import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from models.policy_network import PolicyNetwork


class PPOAgent:
    def __init__(self, input_dim, action_dim, config):
        self.model = PolicyNetwork(input_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.priority_scale = config.get('priority_scale', 1.0)

    def select_action(self, state, graph_data):
        #  输入维度：[1, input_dim], [1, N, N]
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, input_dim]
        graph_data = graph_data if isinstance(graph_data, torch.Tensor) else torch.FloatTensor(graph_data).unsqueeze(0)

        logits, value, priority = self.model(state, graph_data)  # logits: [1, action_dim]
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(), priority.squeeze()

    def update(self, memory):
        states = torch.cat(memory['states'], dim=0)  # [T, input_dim]
        graph_data = torch.cat(memory['graphs'], dim=0)  # [T, N, N]
        actions = torch.tensor(memory['actions'])  # [T]
        rewards = torch.tensor(memory['rewards'], dtype=torch.float32)
        old_log_probs = torch.stack(memory['log_probs'])  # [T]

        returns = self._compute_returns(rewards)  # [T]

        logits, values, priorities = self.model(states, graph_data)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        advantages = returns - values.squeeze()
        priority_weighted_adv = advantages * (1.0 + self.priority_scale * priorities.squeeze())

        ratio = torch.exp(log_probs - old_log_probs.detach())
        surr1 = ratio * priority_weighted_adv
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * priority_weighted_adv

        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values.squeeze(), returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_returns(self, rewards, normalize=True):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
