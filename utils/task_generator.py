import numpy as np
from typing import List, Dict
from models.task import Task


def generate_power_profile() -> Dict[float, float]:
    """
    Generate power profile for different voltage levels
    Returns voltage->power mapping
    """
    # Based on P = αCV²f + Pstatic
    # Values chosen from typical mobile processor characteristics
    return {
        0.8: 0.5,  # Low voltage -> ~0.5W
        1.0: 1.0,  # Nominal -> ~1W
        1.2: 2.0,  # High voltage -> ~2W
    }


def uunifast(n: int, total_util: float) -> List[float]:
    """Generate n utilization values that sum to total_util"""
    if total_util >= n:
        raise ValueError(
            f"Total utilization {total_util} must be less than number of tasks {n}"
        )

    utils = []
    remaining = total_util

    for i in range(n - 1):
        next_util = remaining * (np.random.random() ** (1.0 / (n - i)))
        util = remaining - next_util
        # Ensure no single utilization exceeds 1
        if util > 1.0:
            util = 0.9  # Cap at 90%
            next_util = remaining - util
        utils.append(util)
        remaining = next_util

    utils.append(remaining)

    # Final validation
    if any(u > 1.0 for u in utils):
        return uunifast(n, total_util)

    return utils


def generate_periodic_tasks(
    n: int,
    total_utilization: float,
    min_period_bound: int = 40,
    max_period_bound: int = 200,
) -> List[Task]:
    """Generate n periodic tasks with bounded utilization"""
    if total_utilization >= n:
        raise ValueError("Total utilization must be less than number of tasks")

    utils = uunifast(n, total_utilization)
    tasks = []

    for i, util in enumerate(utils):
        period = np.random.uniform(min_period_bound, max_period_bound)
        wcet = util * period
        while period < wcet:
            period = np.random.uniform(min_period_bound, max_period_bound)
            wcet = util * period

        tasks.append(
            Task(
                id=i,
                wcet=wcet,
                period=period,
                deadline=period,
                power_profile=generate_power_profile(),
            )
        )

    # Validate final task set
    for task in tasks:
        if task.utilization > 1.0:
            print("Invalid task set, regenerating...")
            return generate_periodic_tasks(n, total_utilization)

    if sum(task.utilization for task in tasks) > total_utilization:
        print("Invalid task set, regenerating...")
        return generate_periodic_tasks(n, total_utilization)

    return tasks


def test_task_generation():
    # Generate multiple task sets
    for util in [0.4, 0.6, 0.8]:
        tasks = generate_periodic_tasks(n=10, total_utilization=util)
        print(f"\nTest with utilization {util}:")
        print_task_stats(tasks)
        plot_task_utilizations(tasks)


if __name__ == "__main__":
    test_task_generation()