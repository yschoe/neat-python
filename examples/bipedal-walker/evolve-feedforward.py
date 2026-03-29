"""\
Feed-forward BipedalWalker-v3 control example using Gymnasium.

This example is structured similarly to examples/lunar-lander/evolve-feedforward.py and
produces the same kinds of visual artifacts:

* Fitness curve over generations
* Species size stack plot
* Network diagrams (full and pruned) of the winning genome
"""

import argparse
import copy
import multiprocessing
import os
import pickle

import gymnasium as gym
import neat
import visualize

# Evaluation parameters.
runs_per_net = 1
max_steps = 2000


def eval_genome(genome, config):
    """Evaluate a single genome on the BipedalWalker-v3 environment."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(runs_per_net):
        # Create a fresh environment for each run (no rendering during training).
        env = gym.make("BipedalWalker-v3")
        observation, info = env.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            # Network outputs four continuous action values in [-1, 1].
            # With tanh activations (see config), the raw outputs are already
            # in a good range for the BipedalWalker action space.
            action = net.activate(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        env.close()
        fitnesses.append(total_reward)

    # Use the average reward over runs as the fitness.
    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def get_node_names():
    # BipedalWalker-v3 observations are a 24-dimensional vector that includes
    # hull angle/velocity, joint angles/velocities, leg contact, and LIDAR
    # measurements. For brevity, we group them into coarse labels here.
    return {
        # Example grouping of observation components (indices are illustrative):
        -1: "hull_angle",
        -2: "hull_ang_vel",
        -3: "vel_x",
        -4: "vel_y",
        -5: "hip_1",
        -6: "knee_1",
        -7: "hip_2",
        -8: "knee_2",
        # Remaining inputs (-9 .. -24) are left unnamed for clarity.
        0: "motor_hip_1",
        1: "motor_knee_1",
        2: "motor_hip_2",
        3: "motor_knee_2",
    }


def save_run_artifacts(output_dir, config, genome, stats, node_names, view):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "winner-feedforward.pickle"), "wb") as f:
        pickle.dump(genome, f)

    visualize.plot_stats(
        stats,
        ylog=False,
        view=view,
        filename=os.path.join(output_dir, "feedforward-fitness.svg"),
    )
    visualize.plot_species(
        stats,
        view=view,
        filename=os.path.join(output_dir, "feedforward-speciation.svg"),
    )

    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, "winner-feedforward.gv"),
    )
    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, "winner-feedforward-pruned.gv"),
        prune_unused=True,
    )


class SnapshotReporter(neat.reporting.BaseReporter):
    def __init__(self, snapshot_interval, config, stats, node_names):
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.config = config
        self.stats = stats
        self.node_names = node_names
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        completed_generation = self.generation + 1
        if completed_generation % self.snapshot_interval != 0:
            return

        snapshot_dir = f"snapshot-{completed_generation:05d}"
        save_run_artifacts(
            snapshot_dir,
            self.config,
            copy.deepcopy(best_genome),
            self.stats,
            self.node_names,
            view=False,
        )
        print("\n" + "=" * 72)
        print(
            f" SNAPSHOT SAVED: generation {completed_generation:05d} -> "
            f"{os.path.abspath(snapshot_dir)}"
        )
        print("=" * 72 + "\n")


def run(config_file):
    local_dir = os.path.dirname(__file__)
    config_basename = os.path.basename(config_file)
    run_dir = os.path.join(local_dir, f"exp-{config_basename}")
    os.makedirs(run_dir, exist_ok=True)

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        node_names = get_node_names()
        snapshot_interval = getattr(config, "snapshot_interval", 100)
        p.add_reporter(SnapshotReporter(snapshot_interval, config, stats, node_names))
        # Periodic checkpoints, similar to other examples.
        p.add_reporter(neat.Checkpointer(10))

        # Use parallel evaluation across available CPU cores.
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

        # Run until solution or fitness threshold is reached (see config).
        winner = p.run(pe.evaluate, 300)

        # Display the winning genome.
        print(f"\nRun directory: {run_dir}")
        print(f"\nBest genome:\n{winner!s}")
        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)

    return winner, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BipedalWalker feed-forward evolution with a chosen config file."
    )
    parser.add_argument(
        "config_filename",
        nargs="?",
        default="config-feedforward",
        help="Config file name relative to this script, or an absolute path.",
    )
    args = parser.parse_args()

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    if os.path.isabs(args.config_filename):
        config_path = args.config_filename
    else:
        config_path = os.path.join(local_dir, args.config_filename)
    run(config_path)
