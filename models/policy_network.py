import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()

        # 假设 task_vector 的维度是 26，因此第一个线性层的输入维度应该是 26
        self.task_encoder = nn.Linear(26, 64)  # input_dim 应该是 26 或其他合适的值
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)  # 输出的维度为 action_dim

    def forward(self, state, graph_data):
        task_feat = self.task_encoder(state)  # [B, 64]
        x = F.relu(self.fc1(task_feat))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        value = logits.mean(dim=-1)  # 假设价值是 logits 的均值
        priority = torch.zeros_like(value)  # 可以根据需要调整 priority 的计算方式
        return logits, value, priority
