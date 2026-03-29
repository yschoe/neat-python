"""
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
"""

import multiprocessing
import os
import pickle
import argparse
import copy

import cart_pole
import neat
import visualize

runs_per_net = 5
simulation_seconds = 60.0


# Use the CTRNN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config)

    fitnesses = []
    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()
        net.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, sim.time_step, sim.time_step)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)

        # print("{0} fitness {1}".format(net, fitness))

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def get_node_names():
    return {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}


def save_run_artifacts(output_dir, config, genome, stats, node_names, view):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'winner-ctrnn'), 'wb') as f:
        pickle.dump(genome, f)

    visualize.plot_stats(
        stats,
        ylog=True,
        view=view,
        filename=os.path.join(output_dir, "ctrnn-fitness.svg"),
    )
    visualize.plot_species(
        stats,
        view=view,
        filename=os.path.join(output_dir, "ctrnn-speciation.svg"),
    )
    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, "winner-ctrnn.gv"),
    )
    visualize.draw_net(
        config,
        genome,
        view=view,
        node_names=node_names,
        filename=os.path.join(output_dir, "winner-ctrnn-pruned.gv"),
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


def run(config_filename='config-ctrnn'):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
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
        winner = pop.run(pe.evaluate)

        print(f"Run directory: {run_dir}")
        print(winner)
        save_run_artifacts(".", config, winner, stats, node_names, view=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run single-pole balancing (CTRNN) with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config-ctrnn',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
