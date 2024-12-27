from utils.metrics import plot_system_metrics, print_task_table
from configs import DVFSConfig
from agents import DVFSEnvironment, train_dvfs_agent
from utils.task_generator import generate_periodic_tasks
import numpy as np
import os
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)


def evaluate_agent_performance(env, agent, episodes=5):
    """Evaluate agent performance over multiple episodes"""
    total_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def collect_episode_data(env, agent):
    """Fixed episode data collection"""
    obs, _ = env.reset()
    done = False
    episode_data = {"powers": [], "temperatures": [], "utils": [], "assignments": []}

    # Track all 50 tasks
    while not done and len(episode_data["assignments"]) < len(env.tasks):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Record step data
        episode_data["powers"].append(info["powers"])
        episode_data["temperatures"].append(info["temperatures"])
        episode_data["utils"].append(info["core_utils"])
        episode_data["assignments"].append(info["assignments"])

        # Update task objects
        current_task_idx = env.current_task_idx - 1  # Previous task
        if current_task_idx >= 0:
            task = env.tasks[current_task_idx]
            task.assigned_core = info["assignments"][current_task_idx]
            task.assigned_frequency = info["frequencies"][current_task_idx]

        # Force termination if stuck
        if len(episode_data["assignments"]) >= env.config.MAX_EPISODE_STEPS:
            print(f"Warning: Episode exceeded max steps")
            break

    return episode_data


def evaluate_scenario(num_cores: int, util_per_core: float):
    # Create checkpoints directory
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup model name
    model_name = f"{num_cores}C_{util_per_core}U"
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    total_util = util_per_core * num_cores

    try:
        tasks = generate_periodic_tasks(
            n=100,
            total_utilization=total_util,
        )

    except ValueError as e:
        print(f"Failed to generate feasible task set: {e}")
        return None

    config = DVFSConfig(NUM_CORES=num_cores, MAX_EPISODE_STEPS=1000)
    env = DVFSEnvironment(tasks, config)

    print(
        f"\nTraining agent for {num_cores} cores with {util_per_core} utilization per core"
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10k steps
        save_path=checkpoint_path,
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_env = DVFSEnvironment(tasks.copy(), config)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_path,
        log_path=checkpoint_path,
        eval_freq=2048,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=1
        ),
    )

    callbacks = [checkpoint_callback, eval_callback]

    agent = train_dvfs_agent(env, steps=200_000, checkpoint_callback=callbacks)

    # Save final model
    agent.save(os.path.join(checkpoint_path, f"{model_name}_final"))

    # Evaluation and plotting
    episode_data = collect_episode_data(env, agent)
    plot_system_metrics(env, episode_data, name=f"{num_cores}C_{util_per_core}U")
    print_task_table(tasks, name=f"{num_cores}C_{util_per_core}U")

    return agent


def main():
    # scenarios = {"cores": [8, 16, 32], "utils": [0.25, 0.5, 0.75, 1.0]}
    scenarios = {"cores": [8, 16], "utils": [1.0]}

    results = {}
    for cores in scenarios["cores"]:
        results[cores] = {}
        for util in scenarios["utils"]:
            print(f"\nEvaluating scenario: {cores} cores, {util} utilization")
            agent = evaluate_scenario(cores, util)
            results[cores][util] = agent

    return results


if __name__ == "__main__":
    main()
