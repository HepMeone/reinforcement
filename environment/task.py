class Task:
    def __init__(self, task_id, task_type, processing_time, deadline, priority):
        self.task_id = task_id                    # 任务唯一标识
        self.task_type = task_type                # 任务类型（影响资源匹配）
        self.processing_time = processing_time    # 所需处理时间
        self.deadline = deadline                  # 截止时间
        self.priority = priority                  # 任务优先级

        self.wait_time = 0                        # 累积等待时间
        self.start_time = None                    # 实际开始时间
        self.finish_time = None                   # 实际完成时间

        self.department_id = None                 # 所属部门编号（用于跨部门调度）
        self.assigned_machine_id = None           # 被分配的机器 ID
        self.assigned_worker_id = None            # 被分配的工人 ID

        self.is_done = False                      # 是否已完成
        self.delayed_time = 0                     # 超过 deadline 的时长（惩罚用）
        self.transport_cost = 0                   # 跨部门调度的运输成本（奖励函数用）
