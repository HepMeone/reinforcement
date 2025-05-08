import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from environment.scheduling_env import ManufacturingEnv
from models.ppo import PPOAgent

class Trainer:
    def __init__(self, config):
        self.env = ManufacturingEnv(config)
        self.agent = PPOAgent(config['input_dim'], config['action_dim'], config)
        self.epochs = config['epochs']
        self.max_steps = config['max_steps']
        self.log_file = config.get('log_file', 'logs/training_log.txt')

    def train(self):
        for epoch in range(self.epochs):
            state = self.env.reset()
            memory = defaultdict(list)
            total_reward = 0

            for step in range(self.max_steps):
                task_state = state['vector']

                # 将 resource_graph 转为邻接矩阵（用于神经网络输入）
                graph = state['resource_graph']
                adj_matrix = nx.to_numpy_array(graph)
                graph_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)  # [1, N, N]

                # 选择动作
                action, log_prob, value, priority = self.agent.select_action(task_state, graph_tensor)

                # 与环境交互
                next_state, reward, done, info = self.env.step(action)

                # 存储训练所需数据
                memory['states'].append(torch.FloatTensor(task_state).unsqueeze(0))
                memory['graphs'].append(graph_tensor)
                memory['actions'].append(torch.tensor(action))
                memory['log_probs'].append(log_prob)
                memory['rewards'].append(reward)

                state = next_state
                total_reward += reward

                if done:
                    break

            # 用 PPO 进行策略更新
            self.agent.update(memory)

            print(f"Epoch {epoch + 1}/{self.epochs} - Total Reward: {total_reward}")
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch + 1}, Reward: {total_reward}\n")
