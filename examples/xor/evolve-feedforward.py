"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import argparse
import os

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
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(eval_genomes, 300)

        # Display the winning genome.
        print(f"\nRun directory: {run_dir}")
        print(f'\nBest genome:\n{winner!s}')

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = winner_net.activate(xi)
            print(f"input {xi!r}, expected output {xo!r}, got {output!r}")

        node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        p.run(eval_genomes, 10)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run XOR feed-forward evolution with a chosen config file.'
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
