import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from models import Task, ThermalModel, PowerModel
from configs import DVFSConfig
from schedulers import EDFScheduler, DVFSController


class DVFSEnvironment(gym.Env):
    def __init__(self, tasks: List[Task], config: DVFSConfig):
        self.tasks = tasks
        self.config = config
        self.num_cores = config.NUM_CORES
        self.thermal_models = [ThermalModel(config) for _ in range(self.num_cores)]
        self.power_model = PowerModel(config)
        self.core_powers = np.zeros(self.num_cores)
        self.core_utils = np.zeros(self.num_cores)
        self.edf_schedulers = [
            EDFScheduler(config.NUM_CORES) for _ in range(config.NUM_CORES)
        ]
        self.dvfs_controller = DVFSController(list(config.VF_PAIRS.values()))

        self.total_utilization = sum(task.scaled_utilization for task in tasks)
        self.target_util_per_core = self.total_utilization / config.NUM_CORES

        # Discrete action space: (task_id -> [core_id, vf_pair_id])
        num_vf_pairs = len(config.VF_PAIRS)
        self.action_space = gym.spaces.Discrete(self.num_cores * num_vf_pairs)

        # State space: [core_utils, normalized_distance (to target), core_temps, task_info]
        obs_dim = (3 * self.num_cores) + 3
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_task_idx = 0
        self.core_assignments = [-1] * len(self.tasks)
        self.task_frequencies = [1.0] * len(self.tasks)

        # Reset task assignments
        for task in self.tasks:
            task.assigned_core = None
            task.assigned_frequency = None

        for model in self.thermal_models:
            model.temperature = self.config.AMBIENT_TEMP

        self.edf_schedulers = [
            EDFScheduler(self.num_cores) for _ in range(self.num_cores)
        ]
        return self._get_state(), {}

    def _get_info(self) -> Dict:
        """Get current environment info"""
        return {
            "temperatures": [m.temperature for m in self.thermal_models],
            "powers": self.core_powers,
            "frequencies": self.task_frequencies,
            "core_utils": self.core_utils,
            "assignments": self.core_assignments,
        }

    def step(self, action):
        if self.current_task_idx >= len(self.tasks):
            print(
                f"Completed assignments: {sum(1 for x in self.core_assignments if x is not None)}/{len(self.tasks)}"
            )
            return (
                self._get_state(),
                self._calculate_final_reward(),
                True,
                False,
                self._get_info(),
            )

        # Decode action and check validity
        core_id = action // len(self.config.VF_PAIRS)
        vf_idx = action % len(self.config.VF_PAIRS)
        current_task = self.tasks[self.current_task_idx]

        # Try to find valid core if current one fails
        original_core = core_id
        for attempt in range(self.num_cores):
            core_id = (original_core + attempt) % self.num_cores
            predicted_util = self.core_utils[core_id] + current_task.scaled_utilization
            if predicted_util <= 1.0:
                break
        else:
            # No valid core found - end episode with failure (large penalty = -10)
            return self._get_state(), -10.0, True, False, self._get_info()

        # Voltage-Frequency pair
        v, f = list(self.config.VF_PAIRS.items())[vf_idx]

        # Assign task
        self.core_assignments[self.current_task_idx] = core_id
        self.task_frequencies[self.current_task_idx] = f
        current_task.assigned_core = core_id
        current_task.assigned_frequency = f

        # Update metrics
        self.core_utils = np.zeros(self.num_cores)
        self.core_powers = np.zeros(self.num_cores)

        for task_idx, (assigned_core, freq) in enumerate(
            zip(self.core_assignments, self.task_frequencies)
        ):
            if assigned_core >= 0:
                task = self.tasks[task_idx]
                if self.core_utils[assigned_core] + task.scaled_utilization > 1.0:
                    return self._get_state(), -10.0, True, False, self._get_info()

                self.core_utils[assigned_core] += task.scaled_utilization
                v = list(self.config.VF_PAIRS.keys())[
                    list(self.config.VF_PAIRS.values()).index(freq)
                ]
                self.core_powers[assigned_core] += self.power_model.calculate_power(
                    v, freq, task.scaled_utilization
                )

        # Update temperatures and calculate reward
        core_temps = np.array(
            [m.update(p) for m, p in zip(self.thermal_models, self.core_powers)]
        )
        reward = self._calculate_step_reward(core_temps, f)

        # Move to next task
        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)

        return self._get_state(), reward, done, False, self._get_info()

    def _get_state(self):
        """Fixed state representation"""
        # Handle end of episode
        if self.current_task_idx >= len(self.tasks):
            # Return zero state when no more tasks
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Current task info
        current_task = self.tasks[self.current_task_idx]
        task_util = current_task.scaled_utilization

        # Core states
        core_utils = self.core_utils.copy()
        distance_from_target = np.abs(core_utils - self.target_util_per_core)

        # Normalize components
        normalized_distance = distance_from_target / self.target_util_per_core
        core_temps = (
            np.array([m.temperature for m in self.thermal_models])
            - self.config.AMBIENT_TEMP
        ) / (self.config.MAX_TEMPERATURE - self.config.AMBIENT_TEMP)

        return np.concatenate(
            [
                core_utils,  # [num_cores]
                normalized_distance,  # [num_cores]
                core_temps,  # [num_cores]
                [task_util],  # [1]
                [self.target_util_per_core],  # [1]
                [self.current_task_idx / len(self.tasks)],  # [1]
            ]
        ).astype(np.float32)

    def _calculate_step_reward(self, core_temps, current_freq):
        reward = 0.0

        # 1. Critical Constraints (-10 to 0)
        if any(temp >= self.config.MAX_TEMPERATURE for temp in core_temps):
            return -10.0

        if any(util > 1.0 for util in self.core_utils):
            return -10.0

        # 2. Primary Objectives (-5 to 0)
        # Utilization balance relative to target
        util_deviation = np.mean(
            [abs(util - self.target_util_per_core) for util in self.core_utils]
        )
        balance_reward = -5.0 * (util_deviation / self.target_util_per_core)
        reward += balance_reward

        # Temperature management with soft threshold
        temp_headroom = [
            (self.config.MAX_TEMPERATURE - temp)
            / (self.config.MAX_TEMPERATURE - self.config.AMBIENT_TEMP)
            for temp in core_temps
        ]
        temp_reward = -3.0 * (1.0 - np.mean(temp_headroom))
        reward += temp_reward

        # 3. Secondary Objectives (-2 to +2)
        # Power efficiency
        power_usage = np.sum(self.core_powers) / (
            self.config.MAX_POWER * self.num_cores
        )
        power_reward = -2.0 * power_usage
        reward += power_reward

        # Performance scaling
        freq_reward = 1.0 * (current_freq / max(self.config.VF_PAIRS.values()))
        reward += freq_reward

        # 4. Progress Bonus (0 to 1)
        completion_bonus = 0.1 * (self.current_task_idx / len(self.tasks))
        reward += completion_bonus

        return reward

    def _calculate_final_reward(self):
        """Final reward with focus on constraints"""
        # Check critical constraints
        if any(util > 1.0 for util in self.core_utils):
            return -20.0

        if any(
            m.temperature >= self.config.MAX_TEMPERATURE for m in self.thermal_models
        ):
            return -20.0

        # Verify schedulability
        for core_id in range(self.num_cores):
            core_tasks = [
                t for t, c in zip(self.tasks, self.core_assignments) if c == core_id
            ]
            if not self.edf_schedulers[core_id].is_schedulable(core_tasks, core_id):
                return -20.0

        # Calculate final score
        util_balance = -5.0 * np.std(self.core_utils)
        power_efficiency = -3.0 * (
            np.sum(self.core_powers) / (self.config.MAX_POWER * self.num_cores)
        )
        temp_management = -2.0 * np.mean(
            [
                (t.temperature - self.config.AMBIENT_TEMP)
                / (self.config.MAX_TEMPERATURE - self.config.AMBIENT_TEMP)
                for t in self.thermal_models
            ]
        )

        return (
            util_balance + power_efficiency + temp_management + 10.0
        )  # Bonus for feasible solution

    # def _calculate_reward(
    #     self, vf_pairs: List[Tuple[float, float]], power: float
    # ) -> float:
    #     """Normalized reward function"""
    #     # Large penalty for overheating
    #     if self.thermal_model.temperature >= self.config.MAX_TEMPERATURE:
    #         return -10.0

    #     # Temperature penalty (exponential)
    #     temp_norm = (self.thermal_model.temperature - self.config.AMBIENT_TEMP) / (
    #         self.config.MAX_TEMPERATURE - self.config.AMBIENT_TEMP
    #     )
    #     temp_penalty = -np.exp(4 * temp_norm)

    #     # Power efficiency (-1 to 0)
    #     power_reward = -power / self.config.MAX_POWER

    #     # Performance (0 to 1)
    #     perf_reward = np.mean([f for _, f in vf_pairs]) / max(
    #         f for _, f in self.config.VF_PAIRS.items()
    #     )

    #     return 0.4 * power_reward + 0.4 * perf_reward + 0.2 * temp_penalty
