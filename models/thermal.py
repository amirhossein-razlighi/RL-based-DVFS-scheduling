import numpy as np
from models.power import PowerModel
from configs.dvfs_config import DVFSConfig


class ThermalModel:
    def __init__(self, config: DVFSConfig):
        self.config = config
        self.temperature = config.AMBIENT_TEMP

    def update(self, power: float, dt: float = 0.1) -> float:
        dT = (
            power * self.config.THERMAL_RESISTANCE
            - (self.temperature - self.config.AMBIENT_TEMP)
        ) / self.config.THERMAL_CAPACITANCE
        self.temperature += dT * dt
        return self.temperature

    def is_safe(self) -> bool:
        return self.temperature <= self.config.MAX_TEMPERATURE
