from utils.task_generator import generate_periodic_tasks, generate_aperiodic_tasks
from utils.visualization import plot_task_utilizations, print_task_stats, plot_timeline
import pandas as pd


def test_periodic_task_generation():
    # Test with different utilizations
    enumerator = 0
    for util in [0.4, 0.6, 0.8]:
        tasks = generate_periodic_tasks(
            n=10, total_utilization=util, min_period_bound=10, max_period_bound=50
        )
        print(f"\nTest with utilization {util}:")
        print_task_stats(tasks)
        plot_task_utilizations(tasks, enumerator)
        plot_timeline(tasks, enumerator)
        enumerator += 1


def test_aperiodic_task_generation():
    """Test aperiodic task generation"""
    simulation_length = 1000
    aperiodic_tasks = generate_aperiodic_tasks(
        n=20,
        total_util=0.3,
        simulation_length=simulation_length,
    )
    df = pd.DataFrame(
        [
            {
                "id": task.id,
                "arrival_time": task.arrival_time,
                "execution_time": task.execution_time,
                "soft_deadline": task.soft_deadline,
                "importance": task.importance,
            }
            for task in aperiodic_tasks
        ]
    )
    print("\nGenerated Aperiodic Tasks:")
    for task in aperiodic_tasks:
        print(
            f"Task {task.id}: arrival={task.arrival_time:.1f}, "
            f"exec={task.execution_time:.1f}, deadline={task.soft_deadline:.1f}, "
            f"importance={task.importance:.2f}"
        )

    df.to_csv("aperiodic_tasks.csv", index=False)


if __name__ == "__main__":
    # test_periodic_task_generation()
    test_aperiodic_task_generation()
