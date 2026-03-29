"""
Test and visualize a trained controller for the inverted double pendulum.
"""

import os
import pickle
import argparse

import gymnasium as gym
import neat


def test_network(net, episodes=10, render=True, camera_distance=4.0):
    """
    Tests a neural network controller on the inverted double pendulum.
    
    Args:
        net: The neural network to test
        episodes: Number of episodes to run
        render: Whether to render the environment
        camera_distance: Distance of camera from the pendulum (higher = more zoomed out)
    """
    fitnesses = []
    
    # Create environment once and reuse it for all episodes
    if render:
        env = gym.make('InvertedDoublePendulum-v5', render_mode='human')
    else:
        env = gym.make('InvertedDoublePendulum-v5')
    
    try:
        for episode in range(episodes):
            observation, info = env.reset()
            
            # Adjust camera distance for better view (only needs to be set once)
            if episode == 0 and render and hasattr(env.unwrapped, 'mujoco_renderer'):
                renderer = env.unwrapped.mujoco_renderer
                if renderer.viewer is not None:
                    renderer.viewer.cam.distance = camera_distance
            
            fitness = 0.0
            step = 0
            
            while True:
                step += 1
                # Get action from network
                action = net.activate(observation)
                
                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward
                
                if terminated or truncated:
                    break
            
            fitnesses.append(fitness)
            print(f"Episode {episode + 1}: steps={step}, fitness={fitness:.2f}")
    
    finally:
        env.close()
    
    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)
    
    print(f"\nResults over {episodes} episodes:")
    print(f"  Average fitness: {avg_fitness:.2f}")
    print(f"  Max fitness: {max_fitness:.2f}")
    print(f"  Min fitness: {min_fitness:.2f}")
    
    return fitnesses


def load_and_test(genome_path, config_path, episodes=10, render=True, camera_distance=4.0):
    """
    Loads a saved genome and tests it.
    
    Args:
        genome_path: Path to the pickled genome file
        config_path: Path to the NEAT config file
        episodes: Number of test episodes
        render: Whether to render the environment
        camera_distance: Distance of camera from the pendulum (higher = more zoomed out)
    """
    # Load the config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Load the genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    print('Loaded genome:')
    print(genome)
    
    # Create the network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Test the network
    return test_network(net, episodes=episodes, render=render, camera_distance=camera_distance)


def resolve_run_dir_and_genome(local_dir, config_path, genome_path=None, snapshot=None):
    if genome_path is not None:
        return os.path.dirname(os.path.abspath(genome_path)) or os.getcwd(), genome_path

    winner_name = 'winner-feedforward.pickle'
    config_basename = os.path.basename(config_path)

    if snapshot is not None:
        run_dir = os.path.join(
            local_dir,
            f'exp-{config_basename}',
            f'snapshot-{snapshot:05d}',
        )
        return run_dir, os.path.join(run_dir, winner_name)

    cwd_genome = os.path.join(os.getcwd(), winner_name)
    if os.path.exists(cwd_genome):
        return os.getcwd(), cwd_genome

    run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    return run_dir, os.path.join(run_dir, winner_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test InvertedDoublePendulum winner with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='config-feedforward',
        help='Config file name relative to this script, or an absolute path.',
    )
    parser.add_argument(
        'genome_path',
        nargs='?',
        default=None,
        help='Optional explicit path to winner genome.',
    )
    parser.add_argument(
        '--snapshot',
        type=int,
        default=None,
        help='Snapshot generation number to load (e.g. 100 loads snapshot-00100).',
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
    
    # Check if genome file exists
    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}")
        print(f"Checked cwd and run directory: {run_dir}")
        print("Please train a network first by running evolve-feedforward.py")
        raise SystemExit(1)
    
    # Test the network
    print(f"Run directory: {run_dir}")
    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=5, render=True, camera_distance=4.0)
