"""
This example produces networks that can remember a variable-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!
"""

import multiprocessing
import os
import random
import argparse
import copy
import pickle

import neat
import visualize

# Maximum length of the test sequence.
max_inputs = 3
# Maximum number of ignored inputs
max_ignore = 1
# Number of random examples each network is tested against.
num_tests = 2 ** (max_inputs + max_ignore + 1)


def test_network(net, input_sequence, num_ignore):
    # Feed input bits to the network with the record bit set enabled and play bit disabled.
    net.reset()
    for s in input_sequence:
        inputs = [s, 1.0, 0.0]
        net.activate(inputs)

    # Feed random inputs to be ignored, with both record and play bits disabled.
    for _ in range(num_ignore):
        inputs = [random.choice((0.0, 1.0)), 0.0, 0.0]
        net.activate(inputs)

    # Enable the play bit and get network output.
    outputs = []
    for s in input_sequence:
        inputs = [random.choice((0.0, 1.0)), 0.0, 1.0]
        outputs.append(net.activate(inputs))

    return outputs


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    error = 0.0
    for _ in range(num_tests):
        num_inputs = random.randint(1, max_inputs)
        num_ignore = random.randint(0, max_ignore)

        # Random sequence of inputs.
        seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]

        net.reset()
        outputs = test_network(net, seq, num_ignore)

        for i, o in zip(seq, outputs):
            error += (o[0] - i) ** 2

    return 1.0 - (error / (max_inputs * num_tests))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def get_node_names():
    return {-1: 'input', -2: 'record', -3: 'play', 0: 'output'}


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
        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        node_names = get_node_names()
        snapshot_interval = getattr(config, "snapshot_interval", 100)
        pop.add_reporter(SnapshotReporter(snapshot_interval, config, stats, node_names))

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = pop.run(pe.evaluate, 1000)

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

            num_inputs = random.randint(1, max_inputs)
            num_ignore = random.randint(0, max_ignore)

            seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]
            winner_net.reset()
            outputs = test_network(winner_net, seq, num_ignore)

            correct = True
            for i, o in zip(seq, outputs):
                print(f"\texpected {i:1.5f} got {o[0]:1.5f}")
                correct = correct and round(o[0]) == i
            print("OK" if correct else "FAIL")
            num_correct += 1 if correct else 0

        print(f"{num_correct} of {num_tests} correct {100.0 * num_correct / num_tests:.2f}%")

        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run memory-variable example with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
