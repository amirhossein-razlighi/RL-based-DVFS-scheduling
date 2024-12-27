import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from models import Task


def calculate_qos_task(task: Task, assigned_freq: float, max_freq: float) -> float:
    """Calculate QoS for single task"""
    return assigned_freq / max_freq


def calculate_qos_system(
    tasks: List[Task], frequencies: List[float], max_freq: float
) -> float:
    """Calculate system-wide QoS"""
    return np.mean([f / max_freq for f in frequencies])


def plot_system_metrics(env, episode_data: Dict, name: str = "System Metrics"):
    """Plot system metrics over time with optimized legends"""
    # Create larger figure for better layout when many cores
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # QoS per task
    tasks_qos = [
        calculate_qos_task(t, f, max(env.config.VF_PAIRS.values()))
        for t, f in zip(env.tasks, env.task_frequencies)
    ]
    ax1.bar(range(len(tasks_qos)), tasks_qos)
    ax1.set_title("QoS per Task")
    ax1.set_xlabel("Task ID")
    ax1.set_ylabel("QoS")

    # Power consumption per core with optimized legend
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

    # Temperature profile with optimized legend
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

    # Optimize legends based on number of cores
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

    plt.tight_layout()
    plt.savefig(f"results/{name}.png", bbox_inches="tight", dpi=300)
    plt.close()


def print_task_table(tasks: List[Task], name: str = "Task Properties"):
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
    df.to_csv(f"results/{name}.csv", index=False)
