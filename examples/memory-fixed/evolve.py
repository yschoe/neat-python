"""
This example produces networks that can remember a fixed-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!

This example also demonstrates the use of a custom activation function.
"""

import math
import os
import random
import argparse
import copy
import pickle

import neat
import visualize


# Demonstration of how to add your own custom activation function.
# This sinc function will be available if my_sinc_function is included in the
# config file activation_options option under the DefaultGenome section.
# Note that sinc is not necessarily useful for this example, it was chosen
# arbitrarily just to demonstrate adding a custom activation function.
def sinc(x):
    return 1.0 if x == 0 else math.sin(x) / x


# Demonstration of how to add your own custom aggregation function.
# This l2norm function will be available if my_l2norm_function is included in the
# config file aggregation_options option under the DefaultGenome section.
# Note that l2norm is not necessarily useful for this example, it was chosen
# arbitrarily just to demonstrate adding a custom aggregation function.
def l2norm(x):
    return (sum(i**2 for i in x))**0.5


# N is the length of the test sequence.
N = 4
# num_tests is the number of random examples each network is tested against.
num_tests = 2 ** (N + 2)


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    error = 0.0
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network with the
        # second input set to zero.
        seq = [random.choice((0.0, 1.0)) for _ in range(N)]
        net.reset()
        for s in seq:
            inputs = [s, 0.0]
            net.activate(inputs)

        # Set the second input to one, and get the network output.
        for s in seq:
            inputs = [0.0, 1.0]
            output = net.activate(inputs)

            error += (round(output[0]) - s) ** 2

    return 4.0 - 4.0 * (error / (N * num_tests))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def get_node_names():
    return {-1: 'input', -2: 'gate', 0: 'output'}


def save_run_artifacts(output_dir, config, genome, stats, node_names, view):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'winner.pickle'), 'wb') as f:
        pickle.dump(genome, f)

    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, 'winner.gv'),
    )
    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        prune_unused=True,
        filename=os.path.join(output_dir, 'winner-pruned.gv'),
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


def run(config_filename='config'):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    if os.path.isabs(config_filename):
        config_path = config_filename
    else:
        config_path = os.path.join(local_dir, config_filename)
    config_basename = os.path.basename(config_path)
    run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    os.makedirs(run_dir, exist_ok=True)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        # Demonstration of saving a configuration back to a text file.
        config.save('test_save_config.txt')

        # Demonstration of how to add your own custom activation function.
        # This sinc function will be available if my_sinc_function is included in the
        # config file activation_options option under the DefaultGenome section.
        config.genome_config.add_activation('my_sinc_function', sinc)

        # Demonstration of how to add your own custom aggregation function.
        # This l2norm function will be available if my_l2norm_function is included in the
        # config file aggregation_options option under the DefaultGenome section.
        config.genome_config.add_aggregation('my_l2norm_function', l2norm)

        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        node_names = get_node_names()
        snapshot_interval = getattr(config, "snapshot_interval", 100)
        pop.add_reporter(SnapshotReporter(snapshot_interval, config, stats, node_names))

        winner = pop.run(eval_genomes, 200)

        # Log statistics.
        stats.save()

        # Show output of the most fit genome against a random input.
        print(f"Run directory: {run_dir}")
        print(f'\nBest genome:\n{winner!s}')
        print('\nOutput:')
        winner_net = neat.nn.RecurrentNetwork.create(winner, config)
        num_correct = 0
        for n in range(num_tests):
            print(f'\nRun {n} output:')
            seq = [random.choice((0.0, 1.0)) for _ in range(N)]
            winner_net.reset()
            for s in seq:
                inputs = [s, 0.0]
                winner_net.activate(inputs)
                print(f'\tseq {inputs}')

            correct = True
            for s in seq:
                output = winner_net.activate([0, 1])
                print(f"\texpected {s:1.5f} got {output[0]:1.5f}")
                correct = correct and round(output[0]) == s
            print("OK" if correct else "FAIL")
            num_correct += 1 if correct else 0

        print(f"{num_correct} of {num_tests} correct {100.0 * num_correct / num_tests:.2f}%")

        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run memory-fixed example with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
