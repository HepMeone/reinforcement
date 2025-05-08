CONFIG = {
    # 环境设置
    'env_name': 'ManufacturingScheduling-v0',   # 自定义环境的注册名（如果用 gym 注册）
    'task_data_path': 'data/task_data.csv',     # 初始任务数据的CSV路径
    'num_departments': 3,                       # 参与调度的部门数
    'departments': ['DeptA', 'DeptB', 'DeptC'], # 各部门名称列表
    'num_machines_per_dept': 2,                 # 每个部门的机器数
    'num_workers_per_dept': 2,                  # 每个部门的工人数
    'transport_time': 2,                        # 跨部门传输任务的耗时（步数）
    'max_tasks': 10,                            # 每轮中任务池的最大任务数量

    # 训练设置
    'max_episode': 1000,                        # 最大训练轮数（episode数量）
    'max_steps_per_episode': 200,               # 每轮最大步数（防止卡住）
    'gamma': 0.99,                              # 奖励折扣因子（PPO核心参数）
    'gae_lambda': 0.95,                         # GAE优势估计的平滑系数
    'clip_param': 0.2,                          # 用于剪裁 advantage（如果有用）
    'eps_clip': 0.2,                            # 策略更新中的裁剪范围（PPO核心）
    'ppo_epochs': 10,                           # 每次更新中 PPO 的迭代次数
    'mini_batch_size': 64,                      # mini-batch 的大小
    'lr': 3e-4,                                  # 学习率
    'entropy_coef': 0.01,                       # 策略熵的损失系数（鼓励探索）
    'value_loss_coef': 0.5,                     # 价值函数损失的权重
    'priority_scale': 2.0,                      # 任务优先级影响调度的比例系数

    # 模型结构
    'input_dim': 15,                            # 状态向量维度（需要与你 state 构造逻辑一致）
    'action_dim': 10,                           # 动作空间大小（即最大可调度任务数量）
    'use_gnn': True,                            # 是否启用 GNN（图神经网络）结构
    'use_lstm': True,                           # 是否启用 LSTM 结构

    # 日志与模型保存路径
    'log_path': 'logs/',                        # tensorboard 或自定义日志输出目录
    'save_model_path': 'logs/ppo_model.pt',     # 训练好的模型保存路径

    'epochs': 100,  # 每轮训练中整体重复几次 PPO 训练周期（可用于非 episode 结构）
    'max_steps': 200,  # 每个 episode 最大步数（与 max_steps_per_episode 含义重复，可统一）

    # 设备
    'device': 'cuda'                            # 使用 'cuda' 或 'cpu'
}
