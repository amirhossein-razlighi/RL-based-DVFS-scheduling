from typing import List, Dict, Optional
from models.task import Task
from models.aperiodic_task import AperiodicTask
from configs.dvfs_config import DVFSConfig
from schedulers.dvfs import DVFSController


class ImprovedTBS:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.last_deadline = 0

    def compute_deadline(self, arrival: float, execution: float) -> float:
        k = max(arrival, self.last_deadline)
        deadline = k + (execution / self.capacity)
        self.last_deadline = k  # Update with k instead of deadline
        return deadline


class OnlineScheduler:
    def __init__(self, periodic_tasks: List[Task], config: DVFSConfig):
        self.config = config
        self.core_utils = [0.0] * config.NUM_CORES

        # Track periodic load per core
        for task in periodic_tasks:
            if task.assigned_core is not None:
                self.core_utils[task.assigned_core] += task.scaled_utilization

        # Initialize per-core TBS
        self.tbs_capacities = [min(0.3, 1.0 - util) for util in self.core_utils]
        print("TBS Capacities per core:", self.tbs_capacities)
        self.tbs = [ImprovedTBS(cap) for cap in self.tbs_capacities]

        self.active_tasks = []
        self.completed_tasks = []
        self.accepted_task_ids = set()

    def schedule_aperiodic(self, task: AperiodicTask) -> Optional[float]:
        if task.id in self.accepted_task_ids:
            return None

        # Try each core with its own TBS
        best_core = None
        best_deadline = float("inf")
        best_freq = None

        for core in range(self.config.NUM_CORES):
            if self.tbs_capacities[core] <= 0:
                continue

            deadline = self.tbs[core].compute_deadline(
                task.arrival_time, task.execution_time
            )

            # Try frequencies
            for v, f in self.config.VF_PAIRS.items():
                scaled_exec = task.execution_time / f
                task_util = scaled_exec / (deadline - task.arrival_time)

                if (
                    self.core_utils[core] + task_util <= 1.0
                    and scaled_exec <= deadline - task.arrival_time
                    and deadline < best_deadline
                ):
                    best_core = core
                    best_deadline = deadline
                    best_freq = f

        if best_core is not None:
            task.assigned_core = best_core
            task.assigned_frequency = best_freq
            task.soft_deadline = best_deadline
            self.core_utils[best_core] += task.execution_time / (
                best_freq * (best_deadline - task.arrival_time)
            )
            self.active_tasks.append(task)
            self.accepted_task_ids.add(task.id)
            return best_deadline

        return None

    def update(self, current_time: float):
        self.current_time = current_time

        completed = []
        still_active = []

        for task in self.active_tasks:
            finish_time = task.arrival_time + task.scaled_execution_time
            if finish_time <= current_time:
                completed.append(task)
                self.core_utils[task.assigned_core] -= task.scaled_execution_time / (
                    task.soft_deadline - task.arrival_time
                )
            else:
                still_active.append(task)

        self.completed_tasks.extend(completed)
        self.active_tasks = still_active

    def calculate_qos(self) -> float:
        all_tasks = self.completed_tasks + self.active_tasks
        if not all_tasks:
            return 0.0

        total_qos = 0.0
        for task in all_tasks:
            finish_time = task.arrival_time + (
                task.execution_time / task.assigned_frequency
            )

            # Full QoS if meeting deadline
            if finish_time <= task.soft_deadline:
                total_qos += task.importance
            else:
                # Linear decay after deadline
                lateness = finish_time - task.soft_deadline
                max_lateness = task.soft_deadline - task.arrival_time
                qos = max(0.0, 1.0 - lateness / max_lateness)
                total_qos += qos * task.importance

        return total_qos / len(all_tasks)
