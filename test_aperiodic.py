import matplotlib.pyplot as plt
import numpy as np
from utils.task_generator import generate_aperiodic_tasks
import os


def visualize_aperiodic_tasks():
    # Generate sample tasks
    tasks = generate_aperiodic_tasks(
        n=5, total_util=0.3, simulation_length=100, min_exec=2, max_exec=10
    )

    # Add artificial deadlines for visualization
    for task in tasks:
        task.soft_deadline = task.arrival_time + task.execution_time * 2

    # Setup plot
    plt.figure(figsize=(15, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))

    # Plot each task
    for i, task in enumerate(tasks):
        # Arrival arrow
        plt.arrow(
            task.arrival_time,
            i,
            0,
            -0.2,
            head_width=2,
            head_length=0.1,
            fc="black",
            ec="black",
        )

        # Execution block
        plt.barh(
            i,
            task.execution_time,
            left=task.arrival_time,
            height=0.5,
            color=colors[i],
            alpha=0.6,
            label=f"Task {i}",
        )

        # Deadline marker (now always present)
        plt.axvline(
            x=task.soft_deadline,
            ymin=(i - 0.5) / len(tasks),
            ymax=(i) / len(tasks),
            color="red",
            linestyle="--",
            alpha=0.5,
        )

    # Configure plot
    plt.yticks(
        range(len(tasks)),
        [
            f"Task {i}\nArrival: {task.arrival_time:.1f}\nExec: {task.execution_time:.1f}\nDeadline: {task.soft_deadline:.1f}"
            for i, task in enumerate(tasks)
        ],
    )
    plt.xlabel("Time")
    plt.title("Aperiodic Tasks Timeline")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    if not os.path.exists("hybrid_results"):
        os.makedirs("hybrid_results")

    plt.tight_layout()
    plt.savefig("hybrid_results/aperiodic_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    visualize_aperiodic_tasks()
