"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import argparse
import os
import copy
import pickle

import neat
import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def get_node_names():
    return {-1: 'A', -2: 'B', 0: 'A XOR B'}


def save_run_artifacts(output_dir, config, genome, stats, node_names, view):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'winner-feedforward.pickle'), 'wb') as f:
        pickle.dump(genome, f)

    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, 'winner-feedforward.gv'),
    )
    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        prune_unused=True,
        filename=os.path.join(output_dir, 'winner-feedforward-pruned.gv'),
    )
    visualize.plot_stats(
        stats,
        ylog=False,
        view=view,
        filename=os.path.join(output_dir, 'avg_fitness.svg'),
    )
    visualize.plot_species(
        stats,
        view=view,
        filename=os.path.join(output_dir, 'speciation.svg'),
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


def run(config_filename='config-feedforward'):
    local_dir = os.path.dirname(__file__)
    if os.path.isabs(config_filename):
        config_path = config_filename
    else:
        config_path = os.path.join(local_dir, config_filename)
    config_basename = os.path.basename(config_path)
    run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    os.makedirs(run_dir, exist_ok=True)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        node_names = get_node_names()
        snapshot_interval = getattr(config, "snapshot_interval", 100)
        p.add_reporter(SnapshotReporter(snapshot_interval, config, stats, node_names))

        # Run until a solution is found.
        winner = p.run(eval_genomes)

        # Display the winning genome.
        print(f"Run directory: {run_dir}")
        print(f'\nBest genome:\n{winner!s}')

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = winner_net.activate(xi)
            print(f"  input {xi!r}, expected output {xo!r}, got {output!r}")
        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run XOR minimal evolution with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config-feedforward',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
