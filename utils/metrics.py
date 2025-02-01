import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from models import Task
from models.aperiodic_task import AperiodicTask
import os


def calculate_qos_task(task: Task, assigned_freq: float, max_freq: float) -> float:
    """Calculate QoS for single task"""
    return assigned_freq / max_freq


def calculate_qos_system(
    tasks: List[Task], frequencies: List[float], max_freq: float
) -> float:
    """Calculate system-wide QoS"""
    return np.mean([f / max_freq for f in frequencies])


def plot_system_metrics(
    env, episode_data: Dict, name: str = "System Metrics", path: str = "hybrid_results"
):
    """Plot system metrics over time"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # QoS per task
    # tasks_qos = [
    #     calculate_qos_task(t, f, max(env.config.VF_PAIRS.values()))
    #     for t, f in zip(env.tasks, env.task_frequencies)
    # ]
    tasks_qos = [
        1.0 for t, f in zip(env.tasks, env.task_frequencies)
    ]  # since they are scheduled and they are hard deadline, they are always 1
    ax1.bar(range(len(tasks_qos)), tasks_qos)
    ax1.set_title("QoS per Task")
    ax1.set_xlabel("Task ID")
    ax1.set_ylabel("QoS")

    # Power consumption per core
    times = range(len(episode_data["powers"]))
    lines = []
    for core in range(env.num_cores):
        line = ax2.plot(
            times, [p[core] for p in episode_data["powers"]], alpha=0.7, linewidth=1
        )
        lines.append(line[0])

    ax2.set_title("Power Consumption Over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Power (W)")

    # Temperature profile
    temp_lines = []
    for core in range(env.num_cores):
        line = ax3.plot(
            times,
            [t[core] for t in episode_data["temperatures"]],
            alpha=0.7,
            linewidth=1,
        )
        temp_lines.append(line[0])

    ax3.set_title("Temperature Profile")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Temperature (°C)")

    # Core utilization
    utils = [env.core_utils[i] for i in range(env.num_cores)]
    ax4.bar(range(env.num_cores), utils)
    ax4.set_title("Core Utilization")
    ax4.set_xlabel("Core ID")
    ax4.set_ylabel("Utilization")

    if env.num_cores > 16:
        # Move legends outside with multiple columns
        ax2.legend(
            lines,
            [f"Core {i}" for i in range(env.num_cores)],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=max(2, env.num_cores // 8),
            fontsize="x-small",
        )

        ax3.legend(
            temp_lines,
            [f"Core {i}" for i in range(env.num_cores)],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=max(2, env.num_cores // 8),
            fontsize="x-small",
        )

        # Adjust layout to make room for legends
        plt.subplots_adjust(right=0.85)
    else:
        # Default legend for fewer cores
        ax2.legend(
            lines,
            [f"Core {i}" for i in range(env.num_cores)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize="small",
        )

        ax3.legend(
            temp_lines,
            [f"Core {i}" for i in range(env.num_cores)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize="small",
        )

    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close()


def print_task_table(
    tasks: List[Task], name: str = "Task Properties", path: str = "hybrid_results"
):
    data = {
        "ID": [t.id for t in tasks],
        "WCET": [t.wcet for t in tasks],
        "Period": [t.period for t in tasks],
        "Deadline": [t.deadline for t in tasks],
        "Utilization": [t.scaled_utilization for t in tasks],  # Changed
        "Assigned Core": [t.assigned_core for t in tasks],
        "Assigned Freq": [t.assigned_frequency for t in tasks],
    }

    df = pd.DataFrame(data)
    print(df.to_string())
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{name}.csv", index=False)


def plot_aperiodic_metrics(
    aperiodic_tasks: List[AperiodicTask],
    accepted_tasks: List[AperiodicTask],
    qos_history: List[float],
    simulation_length: int,
    name: str = "Aperiodic_Metrics",
    path: str = "hybrid_results",
):
    """Plot metrics including frequency assignments"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Response Time Distribution
    response_times = [(t.soft_deadline - t.arrival_time) for t in accepted_tasks]
    ax1.hist(response_times, bins=20, alpha=0.7)
    ax1.set_title("Response Time Distribution")
    ax1.set_xlabel("Response Time")
    ax1.set_ylabel("Count")

    # Acceptance Rate Over Time
    arrivals = np.zeros(simulation_length)
    accepts = np.zeros(simulation_length)
    for t in aperiodic_tasks:
        arrivals[int(t.arrival_time)] += 1
    for t in accepted_tasks:
        accepts[int(t.arrival_time)] += 1

    cumul_rate = np.cumsum(accepts) / np.maximum(1, np.cumsum(arrivals))
    ax2.plot(cumul_rate, label="Acceptance Rate")
    ax2.set_title("Cumulative Acceptance Rate")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Rate")
    ax2.legend()

    # QoS Over Time
    ax3.plot(qos_history, label="QoS")
    ax3.set_title("Quality of Service")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("QoS")
    ax3.legend()

    # Frequency assignments
    freqs = [t.assigned_frequency for t in accepted_tasks]
    ax4.hist(freqs, bins=20, alpha=0.7)
    ax4.set_title("Frequency Assignments")
    ax4.set_xlabel("Frequency (GHz)")
    ax4.set_ylabel("Number of Tasks")

    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close()


def print_aperiodic_table(
    tasks: List[AperiodicTask],
    name: str = "Aperiodic_Tasks",
    path: str = "hybrid_results",
):
    """Print aperiodic task properties with core/frequency assignments"""
    data = {
        "ID": [t.id for t in tasks],
        "Arrival": [t.arrival_time for t in tasks],
        "Execution": [t.execution_time for t in tasks],
        "Deadline": [t.soft_deadline for t in tasks],
        "Importance": [t.importance for t in tasks],
        "Response Time": [
            (t.soft_deadline - t.arrival_time) if t.soft_deadline else None
            for t in tasks
        ],
        "Assigned Core": [t.assigned_core for t in tasks],
        "Assigned Freq": [t.assigned_frequency for t in tasks],
        "Accepted": [t.soft_deadline is not None for t in tasks],
    }

    df = pd.DataFrame(data)
    print(df.to_string())
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{name}.csv", index=False)
