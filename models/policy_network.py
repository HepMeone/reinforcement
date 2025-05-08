# models/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Task state encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 这里我们不知道邻接矩阵大小，所以将第一个 GNN 层延后初始化
        self.gnn_fc1 = None  # lazy init
        self.gnn_fc2 = nn.Linear(64, 32)  # 第二层保持原样

        self.combined_layer = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
        self.priority_head = nn.Linear(64, 1)

    def forward(self, task_vector, graph_data):
        task_feat = self.task_encoder(task_vector)  # [B, 32]

        B, N, _ = graph_data.shape
        graph_flat = graph_data.view(B, -1)         # [B, N*N]

        # Lazy 初始化 GNN 输入层
        if self.gnn_fc1 is None:
            in_dim = N * N
            self.gnn_fc1 = nn.Linear(in_dim, 64).to(graph_data.device)

        x = F.relu(self.gnn_fc1(graph_flat))        # [B, 64]
        gnn_feat = self.gnn_fc2(x)                  # [B, 32]

        combined = torch.cat([task_feat, gnn_feat], dim=-1)  # [B, 64]
        combined = F.relu(self.combined_layer(combined))

        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)
        priority = torch.sigmoid(self.priority_head(combined))

        return policy_logits, value, priority
