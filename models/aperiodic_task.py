from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AperiodicTask:
    id: int
    arrival_time: float  # Arrival time in system
    execution_time: float  # WCET
    soft_deadline: Optional[float]  # Assigned by TBS
    importance: float  # QoS weight (0-1)
    power_profile: Dict[float, float]  # Voltage -> Power mapping
    assigned_core: Optional[int] = None
    assigned_frequency: Optional[float] = None

    @property
    def utilization(self) -> float:
        """Calculate task utilization"""
        if not self.soft_deadline:
            return 0.0
        return self.execution_time / self.soft_deadline

    @property
    def response_time(self) -> Optional[float]:
        """Get response time if task has deadline"""
        if not self.soft_deadline:
            return None
        return self.soft_deadline - self.arrival_time

    @property
    def is_completed(self) -> bool:
        """Check if task completed within deadline"""
        if not self.soft_deadline:
            return False
        return self.soft_deadline >= self.arrival_time + self.execution_time

    @property
    def scaled_execution_time(self) -> float:
        """Get execution time scaled by frequency"""
        if not self.assigned_frequency:
            return self.execution_time
        return self.execution_time / self.assigned_frequency
