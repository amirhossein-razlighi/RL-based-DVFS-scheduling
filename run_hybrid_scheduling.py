from utils.task_generator import generate_periodic_tasks, generate_aperiodic_tasks
from configs.dvfs_config import DVFSConfig
from agents.environment import DVFSEnvironment
from agents.dvfs_agent import train_dvfs_agent
from online.online_agent import OnlineScheduler
from utils.metrics import (
    plot_system_metrics,
    print_task_table,
    print_aperiodic_table,
    plot_aperiodic_metrics,
)
import numpy as np


def print_core_summary(core_tasks, config):
    print("\nCore Assignment Summary:")
    for core in range(config.NUM_CORES):
        tasks = core_tasks[core]
        if tasks:
            avg_freq = np.mean([t.assigned_frequency for t in tasks])
            avg_util = np.mean([t.utilization for t in tasks])
            print(
                f"Core {core}: {len(tasks)} tasks, "
                f"Avg Freq: {avg_freq:.2f}GHz, "
                f"Avg Util: {avg_util:.2f}"
            )
        else:
            print(f"Core {core}: No tasks assigned")


def run_hybrid_scheduling():
    # 1. Setup
    num_cores = 4
    periodic_util = 0.6
    config = DVFSConfig(NUM_CORES=num_cores)

    # 2. Generate and schedule periodic tasks
    periodic_tasks = generate_periodic_tasks(
        n=10, total_utilization=periodic_util * num_cores
    )

    # Train RL agent
    env = DVFSEnvironment(periodic_tasks, config)
    agent = train_dvfs_agent(env, steps=50_000)

    # Get final periodic assignments
    obs, _ = env.reset()
    episode_data = {
        "powers": [],
        "temperatures": [],  # Fix key name
        "utils": [],
        "assignments": [],
    }

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Track metrics with correct keys
        episode_data["powers"].append(info["powers"])
        episode_data["temperatures"].append(info["temperatures"])
        episode_data["utils"].append(info["core_utils"])
        episode_data["assignments"].append(info["assignments"])

        if done:
            break

    # Verify periodic assignments
    print("\nPeriodic Task Assignments:")
    print_task_table(periodic_tasks, "periodic_tasks")
    plot_system_metrics(env, episode_data, "periodic_schedule")

    # 3. Online scheduling
    SIMULATION_LEN = 750
    online_scheduler = OnlineScheduler(periodic_tasks, config)
    aperiodic_tasks = generate_aperiodic_tasks(
        n=5, total_util=0.2, simulation_length=SIMULATION_LEN, min_exec=5, max_exec=20
    )

    # Track online metrics
    online_metrics = {"accepted_tasks": [], "qos": [], "response_times": []}

    # Simulation loop
    current_time = 0
    while current_time < SIMULATION_LEN:
        # Process arrived tasks
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

        # Update and track QoS
        online_scheduler.update(current_time)
        online_metrics["qos"].append(online_scheduler.calculate_qos())
        current_time += 1

    # Final results
    print("\nAperiodic Task Results:")
    print_aperiodic_table(aperiodic_tasks, "aperiodic_tasks")

    print("\nCore Assignment Summary:")
    core_tasks = {i: [] for i in range(config.NUM_CORES)}
    for task in online_metrics["accepted_tasks"]:
        if task.assigned_core is not None:
            core_tasks[task.assigned_core].append(task)

    print_core_summary(core_tasks, config)

    plot_aperiodic_metrics(
        aperiodic_tasks=aperiodic_tasks,
        accepted_tasks=online_metrics["accepted_tasks"],
        qos_history=online_metrics["qos"],
        simulation_length=1000,
        name="aperiodic_schedule",
    )


if __name__ == "__main__":
    run_hybrid_scheduling()
