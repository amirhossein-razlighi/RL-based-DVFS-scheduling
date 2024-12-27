from utils.task_generator import generate_periodic_tasks
from utils.visualization import plot_task_utilizations, print_task_stats, plot_timeline


def test_task_generation():
    # Test with different utilization levels
    enumerator = 0
    for util in [0.4, 0.6, 0.8]:
        tasks = generate_periodic_tasks(n=10, total_utilization=util)
        print(f"\nTest with utilization {util}:")
        print_task_stats(tasks)
        plot_task_utilizations(tasks, enumerator)
        plot_timeline(tasks, enumerator)
        enumerator += 1


if __name__ == "__main__":
    test_task_generation()
