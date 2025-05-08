
# Reinforcement Learning for Multi-Department Task Scheduling

This project implements a reinforcement learning-based task scheduling system for multi-department manufacturing environments, optimized via PPO with GNN-based deadlock prevention and multi-objective reward design.

## Features

- ✅ Multi-resource, multi-department dispatching
- ✅ PPO with LSTM encoder and multi-head outputs
- ✅ Deadlock-aware with dynamic resource graph detection
- ✅ Priority-based reward shaping and curriculum learning
- ✅ Visualization support (Gantt chart, heatmap)

## Directory Structure

See `reinforcement_learning_task_scheduling/` for full project layout including:

- `environment/`: Custom Gym-compatible scheduling environment
- `models/`: Policy network and PPO agent
- `training/`: Training manager
- `utils/`: Reward calculation and visualization tools
- `data/`: Synthetic task dataset generator
- `logs/`: Training logs and visual outputs

## Running Training

```bash
python main.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Author

This project was built for a senior thesis on collaborative manufacturing scheduling.
