"""
A parallel version of XOR using neat.parallel.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-process),
you should see a significant performance improvement by evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

import multiprocessing
import argparse
import copy
import os
import pickle

import neat
import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error


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


def run(config_file):
    local_dir = os.path.dirname(__file__)
    config_basename = os.path.basename(config_file)
    run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    os.makedirs(run_dir, exist_ok=True)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

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

        # Run for up to 300 generations.
        # Use the context manager pattern to ensure proper cleanup of the multiprocessing pool.
        with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as pe:
            winner = p.run(pe.evaluate, 300)

            # Display the winning genome.
            print(f"Run directory: {run_dir}")
            print(f'\nBest genome:\n{winner!s}')

            # Show output of the most fit genome against training data.
            print('\nOutput:')
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            for xi, xo in zip(xor_inputs, xor_outputs):
                output = winner_net.activate(xi)
                print(f"input {xi!r}, expected output {xo!r}, got {output!r}")

            save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run XOR feed-forward parallel evolution with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config-feedforward',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if os.path.isabs(args.config_filename):
        config_path = args.config_filename
    else:
        config_path = os.path.join(local_dir, args.config_filename)
    run(config_path)
