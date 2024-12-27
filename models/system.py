from dataclasses import dataclass
from typing import List


@dataclass
class SystemConfig:
    num_cores: int
    frequencies: List[float]  # Available frequency levels
    max_power_per_core: float
