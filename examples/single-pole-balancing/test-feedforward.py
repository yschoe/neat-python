"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import argparse
import os
import pickle

import neat
from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie

def run(config_filename='config-feedforward'):
    local_dir = os.path.dirname(__file__)
    if os.path.isabs(config_filename):
        config_path = config_filename
    else:
        config_path = os.path.join(local_dir, config_filename)

    config_basename = os.path.basename(config_path)
    exp_run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    cwd_winner = os.path.join(os.getcwd(), 'winner-feedforward')
    exp_winner = os.path.join(exp_run_dir, 'winner-feedforward')
    if os.path.isfile(cwd_winner):
        run_dir = os.getcwd()
        winner_path = cwd_winner
    else:
        run_dir = exp_run_dir
        winner_path = exp_winner
    movie_path = os.path.join(run_dir, 'feedforward-movie.mp4')

    if not os.path.isfile(winner_path):
        raise FileNotFoundError(
            f"Winner file not found: {winner_path}\n"
            f"Checked cwd and run directory: {run_dir}\n"
            f"Run evolve-feedforward.py first with config '{config_filename}'."
        )

    # Load the winner.
    with open(winner_path, 'rb') as f:
        c = pickle.load(f)

    print(f"Run directory: {run_dir}")
    print('Loaded genome:')
    print(c)

    # Load the selected config.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)
    sim = CartPole()

    print()
    print("Initial conditions:")
    print(f"        x = {sim.x:.4f}")
    print(f"    x_dot = {sim.dx:.4f}")
    print(f"    theta = {sim.theta:.4f}")
    print(f"theta_dot = {sim.dtheta:.4f}")
    print()

    # Run the given simulation for up to 120 seconds.
    balance_time = 0.0
    while sim.t < 120.0:
        inputs = sim.get_scaled_state()
        action = net.activate(inputs)

        # Apply action to the simulated cart-pole.
        force = discrete_actuator_force(action)
        sim.step(force)

        # Stop if the network fails to keep the cart within the position or angle limits.
        if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
            break

        balance_time = sim.t

    print(f'Pole balanced for {balance_time:.1f} of 120.0 seconds')

    print()
    print("Final conditions:")
    print(f"        x = {sim.x:.4f}")
    print(f"    x_dot = {sim.dx:.4f}")
    print(f"    theta = {sim.theta:.4f}")
    print(f"theta_dot = {sim.dtheta:.4f}")
    print()
    print("Making movie...")
    make_movie(net, discrete_actuator_force, 15.0, movie_path)
    print(f"Saved movie: {movie_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test a feed-forward winner using a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config-feedforward',
        help='Config file name (relative to this script) or absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
