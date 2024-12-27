from dataclasses import dataclass, field
from typing import Dict


def default_vf_pairs() -> Dict[float, float]:
    return {0.8: 1.0, 1.0: 1.2, 1.2: 1.5}  # V -> F mapping


@dataclass
class DVFSConfig:
    NUM_CORES: int = 4
    VF_PAIRS: Dict[float, float] = field(
        default_factory=lambda: {
            0.8: 1.0,  # V -> F mapping
            1.0: 1.2,
            1.2: 1.5,
        }
    )
    MAX_TEMPERATURE: float = 80.0
    AMBIENT_TEMP: float = 45.0
    THERMAL_RESISTANCE: float = 2.0
    THERMAL_CAPACITANCE: float = 0.1
    MAX_POWER: float = 20.0
    MAX_EPISODE_STEPS: int = 300

    def __post_init__(self):
        # Validate configuration
        if not self.VF_PAIRS:
            raise ValueError("VF_PAIRS cannot be empty")
        if self.MAX_TEMPERATURE <= self.AMBIENT_TEMP:
            raise ValueError("MAX_TEMPERATURE must be greater than AMBIENT_TEMP")
        if self.MAX_POWER <= 0:
            raise ValueError("MAX_POWER must be positive")
