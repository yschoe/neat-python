"""\
Test and visualize the performance of the best genome produced by
examples/lunar-lander/evolve-feedforward.py on the LunarLander-v3 environment.
"""

import os
import pickle
import argparse

import gymnasium as gym
import neat


def run_episodes(net, episodes=3, render=True):
    """Run a few episodes using the provided network and optionally render."""
    if render:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3")

    try:
        rewards = []
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0.0
            step = 0

            while True:
                step += 1
                action_values = net.activate(observation)
                action = max(range(len(action_values)), key=lambda i: action_values[i])

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            print(
                f"Episode {episode + 1}: steps={step}, total_reward={total_reward:.2f}",
            )
    finally:
        env.close()

    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f"\nAverage reward over {len(rewards)} episodes: {avg:.2f}")


def load_and_test(genome_path, config_path, episodes=3, render=True):
    """Load a saved genome and test it on LunarLander-v3."""
    # Load the config.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Load the genome.
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print("Loaded genome:")
    print(genome)

    # Create the network and run episodes.
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    run_episodes(net, episodes=episodes, render=render)


def resolve_run_dir_and_genome(local_dir, config_path, genome_path=None, snapshot=None):
    if genome_path is not None:
        return os.path.dirname(os.path.abspath(genome_path)) or os.getcwd(), genome_path

    winner_name = "winner-feedforward.pickle"
    config_basename = os.path.basename(config_path)

    if snapshot is not None:
        run_dir = os.path.join(
            local_dir,
            f"exp-{config_basename}",
            f"snapshot-{snapshot:05d}",
        )
        return run_dir, os.path.join(run_dir, winner_name)

    cwd_genome = os.path.join(os.getcwd(), winner_name)
    if os.path.exists(cwd_genome):
        return os.getcwd(), cwd_genome

    run_dir = os.path.join(local_dir, f"exp-{config_basename}")
    return run_dir, os.path.join(run_dir, winner_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LunarLander winner with a chosen config file."
    )
    parser.add_argument(
        "config_filename",
        nargs="?",
        default="config-feedforward",
        help="Config file name relative to this script, or an absolute path.",
    )
    parser.add_argument(
        "genome_path",
        nargs="?",
        default=None,
        help="Optional explicit path to winner genome.",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help="Snapshot generation number to load (e.g. 100 loads snapshot-00100).",
    )
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if os.path.isabs(args.config_filename):
        config_path = args.config_filename
    else:
        config_path = os.path.join(local_dir, args.config_filename)

    run_dir, genome_path = resolve_run_dir_and_genome(
        local_dir,
        config_path,
        args.genome_path,
        snapshot=args.snapshot,
    )

    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}")
        print(f"Checked cwd and run directory: {run_dir}")
        print("Please train a network first by running evolve-feedforward.py")
        raise SystemExit(1)

    print(f"Run directory: {run_dir}")
    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=3, render=True)
