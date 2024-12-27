from typing import List
from models.task import Task


class EDFScheduler:
    def __init__(self, num_cores: int):
        self.num_cores = num_cores
        self.tasks_per_core = [[] for _ in range(num_cores)]

    def is_schedulable(self, tasks: List[Task], core_id: int) -> bool:
        # Sum scaled utilizations
        total_util = sum(
            task.scaled_utilization for task in tasks if task.assigned_core == core_id
        )
        return total_util <= 1.0

    def assign_task(self, task: Task, core_id: int) -> bool:
        if core_id < 0 or core_id >= self.num_cores:
            return False

        if self.is_schedulable(self.tasks_per_core[core_id] + [task], core_id):
            self.tasks_per_core[core_id].append(task)
            return True
        return False
