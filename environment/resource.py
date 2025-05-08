class Machine:
    def __init__(self, machine_id, capabilities, department_id):
        self.machine_id = machine_id
        self.id = machine_id
        self.capabilities = capabilities
        self.department_id = department_id
        self.busy = False
        self.remaining_time = 0
        self.current_task = None

class Worker:
    def __init__(self, worker_id, skills, department_id):
        self.worker_id = worker_id
        self.id = worker_id
        self.skills = skills
        self.department_id = department_id
        self.busy = False
