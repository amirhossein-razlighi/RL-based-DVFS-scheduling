from typing import List
from models.task import Task


class DVFSController:
    def __init__(self, available_frequencies: List[float]):
        self.frequencies = sorted(available_frequencies)

    def get_optimal_frequency(self, task: Task, utilization: float) -> float:
        required_freq = utilization * max(self.frequencies)
        valid_freqs = [f for f in self.frequencies if f >= required_freq]
        return min(valid_freqs) if valid_freqs else max(self.frequencies)
