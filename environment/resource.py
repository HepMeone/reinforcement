class Machine:
    def __init__(self, machine_id, capabilities, department_id):
        self.machine_id = machine_id            # 机器唯一标识符
        self.id = machine_id                    # 为兼容图算法等统一的 ID 字段
        self.capabilities = capabilities        # 可处理的任务类型列表
        self.department_id = department_id      # 所属部门编号

        self.busy = False                       # 当前是否忙碌
        self.remaining_time = 0                 # 剩余处理时间（训练时更新）
        self.current_task = None                # 当前正在处理的任务 ID
        self.busy_until = 0                     # 忙碌结束时间戳，用于环境 step 中判断是否可用


class Worker:
    def __init__(self, worker_id, skills, department_id):
        self.worker_id = worker_id              # 工人唯一标识符
        self.id = worker_id                     # 为统一命名，与图表示保持一致
        self.skills = skills                    # 可处理的任务类型列表
        self.department_id = department_id      # 所属部门编号

        self.busy = False                       # 当前是否忙碌
        self.busy_until = 0                     # 忙碌结束时间戳
        self.current_task = None                # 当前正在处理的任务 ID
