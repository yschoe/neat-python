"""\
Feed-forward LunarLander-v3 control example.

This example is structured similarly to examples/xor/evolve-feedforward.py and
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
runs_per_net = 5
max_steps = 1000


def eval_genome(genome, config):
    """Evaluate a single genome on the LunarLander-v3 environment."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(runs_per_net):
        # Create a fresh environment for each run (no rendering during training).
        env = gym.make("LunarLander-v3")
        observation, info = env.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            # Network outputs four action values; take the argmax as the discrete action.
            action_values = net.activate(observation)
            action = max(range(len(action_values)), key=lambda i: action_values[i])

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
    return {
        # Observations
        -1: "x",
        -2: "y",
        -3: "x_dot",
        -4: "y_dot",
        -5: "angle",
        -6: "ang_vel",
        -7: "left_leg",
        -8: "right_leg",
        # Discrete actions
        0: "do_nothing",
        1: "fire_left",
        2: "fire_main",
        3: "fire_right",
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
        winner = p.run(pe.evaluate, 500)

        # Display the winning genome.
        print(f"\nRun directory: {run_dir}")
        print(f"\nBest genome:\n{winner!s}")
        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)

    return winner, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LunarLander feed-forward evolution with a chosen config file."
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
