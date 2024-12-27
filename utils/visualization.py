import matplotlib.pyplot as plt
import numpy as np
from math import lcm
from typing import List
from models.task import Task


def plot_task_utilizations(tasks: List[Task], enumerator: int = 0):
    utils = [task.utilization for task in tasks]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(utils)), utils)
    plt.xlabel("Task ID")
    plt.ylabel("Utilization")
    plt.title(f"Task Utilization Distribution. test {enumerator}")
    plt.grid(True)

    total_util = sum(utils)
    plt.axhline(
        y=total_util / len(utils),
        color="r",
        linestyle="--",
        label=f"Average: {total_util/len(utils):.2f}",
    )
    plt.legend()
    plt.savefig(f"task_utilization_{enumerator}.png")
    plt.close()


def print_task_stats(tasks: List[Task]):
    utils = [task.utilization for task in tasks]
    wcets = [task.wcet for task in tasks]
    periods = [task.period for task in tasks]

    print("Task Set Statistics:")
    print(f"Total Utilization: {sum(utils):.3f}")
    print(f"Number of Tasks: {len(tasks)}")
    print(f"Average WCET: {np.mean(wcets):.2f}")
    print(f"Average Period: {np.mean(periods):.2f}")


def calculate_hyperperiod(periods: List[float], max_length: float = 100) -> float:
    """Calculate bounded hyperperiod"""
    # Round periods to one decimal place
    int_periods = [int(period * 10) for period in periods]
    try:
        hyper = lcm(*int_periods) / 10
        return min(hyper, max_length)
    except TypeError:
        return min(max(periods) * 2, max_length)


def plot_timeline(tasks: List[Task], enumerator: int = 0):
    """Plot timeline visualization of periodic tasks."""
    try:
        # Calculate bounded hyperperiod
        periods = [task.period for task in tasks]
        hyperperiod = calculate_hyperperiod(periods)

        # Setup plot
        plt.figure(figsize=(15, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))

        # For each task
        for idx, (task, color) in enumerate(zip(tasks, colors)):
            # Calculate release times within hyperperiod
            releases = np.arange(0, hyperperiod, task.period)

            # Plot task instances
            for release in releases:
                # Release arrow
                plt.arrow(
                    release,
                    idx,
                    0,
                    -0.2,
                    head_width=0.5,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )

                # Execution block
                plt.gca().add_patch(
                    plt.Rectangle(
                        (release, idx - 0.25),
                        min(task.wcet, task.period),  # Limit WCET visualization
                        0.5,
                        facecolor=color,
                        alpha=0.5,
                    )
                )

                # Deadline marker
                plt.plot(
                    [release + task.deadline, release + task.deadline],
                    [idx - 0.3, idx + 0.3],
                    "r--",
                    linewidth=1,
                )

        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.title(f"Task Timeline (up to time: {hyperperiod:.1f})")
        plt.xlabel("Time")
        plt.ylabel("Task ID")
        plt.yticks(
            range(len(tasks)), [f"T{task.id} (P={task.period:.1f})" for task in tasks]
        )

        plt.xlim(-1, hyperperiod + 1)
        plt.ylim(-1, len(tasks))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting timeline: {e}")
        raise
