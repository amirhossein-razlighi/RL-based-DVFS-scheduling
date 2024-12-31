from utils.task_generator import generate_periodic_tasks
from configs.dvfs_config import DVFSConfig
from agents.environment import DVFSEnvironment
from agents.dvfs_agent import train_dvfs_agent


def evaluate_agent(env: DVFSEnvironment, agent, num_episodes: int = 5):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1}")
        print("-" * 50)

        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            print(f"Step {step_count}:")
            for core in range(env.num_cores):
                print(f"Core {core}:")
                print(f"  Temperature: {info['temperatures'][core]:.2f}°C")
                print(f"  Power: {info['powers'][core]:.2f}W")
                print(f"  Utilization: {info['core_utils'][core]:.2f}")
            print(f"Current Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"\nEpisode ended after {step_count} steps")
                print(f"Total Reward: {episode_reward:.2f}")
                print("Final State:")
                for core in range(env.num_cores):
                    print(f"Core {core}:")
                    print(f"  Temperature: {info['temperatures'][core]:.2f}°C")
                    print(f"  Power: {info['powers'][core]:.2f}W")
                    print(
                        f"  Tasks: {[i for i, c in enumerate(info['assignments']) if c == core]}"
                    )
                break


def main():
    SEED = 42
    tasks = generate_periodic_tasks(n=50, total_utilization=0.7 * 4)
    config = DVFSConfig(NUM_CORES=4)
    env = DVFSEnvironment(tasks, config)

    agent = train_dvfs_agent(env, steps=50000)
    evaluate_agent(env, agent)


if __name__ == "__main__":
    main()
