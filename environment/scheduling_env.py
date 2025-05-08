import gym
import numpy as np
import networkx as nx
from gym import spaces
from environment.task import Task
from environment.resource import Machine, Worker

class ManufacturingEnv(gym.Env):
    def __init__(self, config):
        super(ManufacturingEnv, self).__init__()
        self.config = config
        self.current_time = 0
        self.tasks = []
        self.machines = []
        self.workers = []
        self.departments = config['departments']
        self.transport_time = config['transport_time']
        self.max_tasks = config['max_tasks']
        self.num_departments = config['num_departments']
        self.num_machines_per_dept = config['num_machines_per_dept']
        self.num_workers_per_dept = config['num_workers_per_dept']

        self.action_space = spaces.Discrete(self.max_tasks * self.num_departments)
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        self.resource_graph = nx.DiGraph()

        self.reset()

    def reset(self):
        self.current_time = 0
        self.tasks = []
        self.machines = []
        self.workers = []
        self.resource_graph.clear()

        self._generate_resources()
        self._generate_initial_tasks()

        obs_vector = self._get_obs()
        graph_data = self._get_resource_graph()
        return {
            'vector': obs_vector,
            'resource_graph': graph_data
        }

    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.current_time += 1


        task_idx = action % self.max_tasks
        task = self.tasks[task_idx]

        if task.is_done:
            reward -= 1  # 负面奖励：调度已完成的任务
        else:
            machine = self._find_available_machine(task.task_type)
            worker = self._find_available_worker(task.task_type)

            if machine and worker:
                task.start_time = self.current_time
                task.finish_time = self.current_time + task.processing_time
                task.is_done = True

                machine.busy_until = task.finish_time
                worker.busy_until = task.finish_time

                reward += 10  # 正面奖励：成功调度一个任务

                if task.finish_time <= task.deadline:
                    reward += 5  # 提前完成奖励
                else:
                    reward -= (task.finish_time - task.deadline) * 0.5  # 超期惩罚
            else:
                reward -= 2  # 没有可用资源

        if all(t.is_done for t in self.tasks):
            done = True

        if self._check_deadlock():
            reward -= 20
            done = True
            info['deadlock'] = True

        obs_vector = self._get_obs()
        graph_data = self._get_resource_graph()
        return {
            'vector': obs_vector,
            'resource_graph': graph_data
        }, reward, done, info

    def _generate_resources(self):
        for d_id in range(self.num_departments):
            for m in range(self.num_machines_per_dept):
                machine = Machine(f"M{d_id}_{m}", capabilities=[0, 1], department_id=d_id)
                machine.busy_until = 0
                self.machines.append(machine)
                self.resource_graph.add_node(machine.id, type='machine', dept=d_id)

            for w in range(self.num_workers_per_dept):
                worker = Worker(f"W{d_id}_{w}", skills=[0, 1], department_id=d_id)
                worker.busy_until = 0
                self.workers.append(worker)
                self.resource_graph.add_node(worker.id, type='worker', dept=d_id)

    def _generate_initial_tasks(self):
        for i in range(self.max_tasks):
            t = Task(
                task_id=i,
                task_type=i % 2,
                processing_time=5 + (i % 3),
                deadline=10 + i,
                priority=i % 3
            )
            t.is_done = False
            self.tasks.append(t)
            self.resource_graph.add_node(f"T{i}", type='task', priority=t.priority)

            for machine in self.machines:
                self.resource_graph.add_edge(f"T{i}", machine.id)
            for worker in self.workers:
                self.resource_graph.add_edge(f"T{i}", worker.id)

    def _get_obs(self):
        obs = []

        for task in self.tasks[:5]:
            obs.append(task.processing_time / 10.0)
            obs.append(task.deadline / 100.0)
            obs.append(task.priority / 3.0)

        while len(obs) < 15:
            obs.append(0.0)

        return np.array(obs, dtype=np.float32)

    def _get_resource_graph(self):
        return self.resource_graph

    def _check_deadlock(self):
        try:
            cycle = nx.find_cycle(self.resource_graph, orientation="original")
            return True if cycle else False
        except nx.NetworkXNoCycle:
            return False

    def _find_available_machine(self, task_type):
        for m in self.machines:
            if task_type in m.capabilities and self.current_time >= m.busy_until:
                return m
        return None

    def _find_available_worker(self, task_type):
        for w in self.workers:
            if task_type in w.skills and self.current_time >= w.busy_until:
                return w
        return None
