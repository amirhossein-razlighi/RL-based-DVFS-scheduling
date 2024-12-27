import numpy as np
from configs.dvfs_config import DVFSConfig


class PowerModel:
    def __init__(self, config: DVFSConfig):
        self.config = config

    def calculate_power(
        self, voltage: float, frequency: float, utilization: float
    ) -> float:
        # Increased power scaling
        dynamic_power = 3.0 * utilization * voltage**2 * frequency
        static_power = 0.5 * voltage**2
        return dynamic_power + static_power
