from utils.metrics import (
    plot_system_metrics,
    print_task_table,
    print_aperiodic_table,
    plot_aperiodic_metrics,
)
from configs.dvfs_config import DVFSConfig
from agents.environment import DVFSEnvironment
from agents.dvfs_agent import train_dvfs_agent
from utils.task_generator import generate_periodic_tasks, generate_aperiodic_tasks
from online.online_agent import OnlineScheduler
import numpy as np
import os
import pandas as pd
from datetime import datetime


def evaluate_hybrid_scenario(num_cores: int, periodic_util: float):
    """Evaluate single hybrid scheduling scenario"""

    # Create scenario directory
    scenario_name = f"{num_cores}C_{periodic_util}U"
    scenario_dir = os.path.join("hybrid_results", scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # 1. Setup and Periodic Scheduling
    config = DVFSConfig(NUM_CORES=num_cores)
    periodic_tasks = generate_periodic_tasks(
        n=num_cores * 2,  # Scale tasks with cores
        total_utilization=periodic_util * num_cores,
    )

    # Train RL agent
    env = DVFSEnvironment(periodic_tasks, config)
    agent = train_dvfs_agent(env, steps=50_000)

    # Get periodic assignments
    obs, _ = env.reset()
    episode_data = {"powers": [], "temperatures": [], "assignments": []}

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        for key in episode_data:
            episode_data[key].append(info[key])

        if done:
            break

    # Save periodic results
    print_task_table(periodic_tasks, path=os.path.join(scenario_dir, "periodic_tasks"))
    plot_system_metrics(
        env, episode_data, path=os.path.join(scenario_dir, "periodic_schedule")
    )

    # 2. Online Aperiodic Scheduling
    online_scheduler = OnlineScheduler(periodic_tasks, config)
    simulation_length = 1000

    aperiodic_tasks = generate_aperiodic_tasks(
        n=num_cores,  # Scale with cores
        total_util=0.2,  # Fixed moderate utilization
        simulation_length=simulation_length,
        min_exec=5,
        max_exec=20,
    )

    # Track metrics
    online_metrics = {
        "accepted_tasks": [],
        "qos": [],
        "response_times": [],
        "core_utils": [[] for _ in range(num_cores)],
    }

    # Simulation loop
    current_time = 0
    while current_time < simulation_length:
        arrived_tasks = [
            task
            for task in aperiodic_tasks
            if task.arrival_time <= current_time
            and task.id not in online_scheduler.accepted_task_ids
        ]

        for task in arrived_tasks:
            deadline = online_scheduler.schedule_aperiodic(task)
            if deadline:
                online_metrics["accepted_tasks"].append(task)
                online_metrics["response_times"].append(deadline - task.arrival_time)

        online_scheduler.update(current_time)
        online_metrics["qos"].append(online_scheduler.calculate_qos())
        for core in range(num_cores):
            online_metrics["core_utils"][core].append(online_scheduler.core_utils[core])

        current_time += 1

    # Save aperiodic results
    print_aperiodic_table(
        aperiodic_tasks, path=os.path.join(scenario_dir, "aperiodic_tasks")
    )
    plot_aperiodic_metrics(
        aperiodic_tasks=aperiodic_tasks,
        accepted_tasks=online_metrics["accepted_tasks"],
        qos_history=online_metrics["qos"],
        simulation_length=simulation_length,
        path=os.path.join(scenario_dir, "aperiodic_metrics"),
    )

    # Collect scenario metrics
    metrics = {
        "num_cores": num_cores,
        "periodic_util": periodic_util,
        "periodic_tasks": len(periodic_tasks),
        "aperiodic_tasks": len(aperiodic_tasks),
        "accepted_aperiodic": len(online_metrics["accepted_tasks"]),
        "avg_qos": np.mean(online_metrics["qos"]),
        "avg_response_time": np.mean(online_metrics["response_times"]),
        "avg_power": np.mean(episode_data["powers"]),
        "max_temp": np.max(episode_data["temperatures"]),
    }

    return metrics


def main():
    # Create results directory
    os.makedirs("hybrid_results", exist_ok=True)

    # Define scenarios
    scenarios = {"cores": [8, 16, 32], "utils": [0.25, 0.5, 0.75]}

    # Run scenarios and collect results
    results = []
    for cores in scenarios["cores"]:
        for util in scenarios["utils"]:
            print(f"\nEvaluating scenario: {cores} cores, {util} utilization")
            metrics = evaluate_hybrid_scenario(cores, util)
            results.append(metrics)

    # Save summary results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"hybrid_results/summary_{timestamp}.csv", index=False)

    # Print summary
    print("\nScenario Summary:")
    print(df.to_string())


if __name__ == "__main__":
    main()
