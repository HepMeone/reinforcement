
def compute_reward(task, delay, transport_time, deadlock_occurred, idle_time):
    reward = 0
    reward += 10 * task['priority'] if task['completed'] else 0
    reward -= delay * (1 + task['priority'])  # 延迟惩罚
    reward -= 1 * transport_time             # 运输成本惩罚
    reward -= 20 if deadlock_occurred else 0 # 死锁惩罚
    reward += 0.1 * idle_time                # 设备健康奖励
    return reward
