from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Task:
    id: int
    wcet: float  # Worst-case execution time
    period: float
    deadline: float
    power_profile: Dict[float, float]  # frequency -> power mapping
    assigned_core: Optional[int] = None
    assigned_frequency: Optional[float] = None

    @property
    def scaled_wcet(self) -> float:
        # WCET scales inversely with frequency
        if self.assigned_frequency:
            return self.wcet / self.assigned_frequency
        return self.wcet

    @property
    def scaled_utilization(self) -> float:
        # Utilization = WCET/Period
        # Higher frequency -> Lower utilization
        if self.assigned_frequency:
            return self.utilization / self.assigned_frequency
        return self.utilization

    @property
    def utilization(self) -> float:
        return self.wcet / self.period
