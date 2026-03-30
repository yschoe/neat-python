"""\
Evolve a feed-forward navigation controller in a 2D arena.

The agent receives target-relative inputs and outputs thrust/turn commands.
Artifacts are written under exp-<config-filename>/, matching other examples.
"""

import argparse
import copy
import math
import multiprocessing
import os
import pickle
import random
from functools import partial

import neat
import visualize


ARENA_SIZE = 300.0
MAX_STEPS = 600
TARGET_RADIUS = 10.0
AGENT_SPEED = 3.0
TURN_SCALE = 0.2
EVAL_EPISODES = 5


class Barrier:
    """Line-segment barrier placed between start and target."""

    def __init__(self, start_x, start_y, target_x, target_y, length=100.0):
        mid_x = 0.5 * (start_x + target_x)
        mid_y = 0.5 * (start_y + target_y)

        dx = target_x - start_x
        dy = target_y - start_y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            perp_dx, perp_dy = 1.0, 0.0
        else:
            perp_dx, perp_dy = -dy / norm, dx / norm

        half = 0.5 * float(length)
        self.x1 = mid_x + perp_dx * half
        self.y1 = mid_y + perp_dy * half
        self.x2 = mid_x - perp_dx * half
        self.y2 = mid_y - perp_dy * half
        self.width = 5.0

    def check_collision(self, x, y):
        line_len = math.hypot(self.x2 - self.x1, self.y2 - self.y1)
        if line_len < 1e-9:
            return False

        dir_x = (self.x2 - self.x1) / line_len
        dir_y = (self.y2 - self.y1) / line_len

        rel_x = x - self.x1
        rel_y = y - self.y1
        t = max(0.0, min(line_len, dir_x * rel_x + dir_y * rel_y))

        near_x = self.x1 + t * dir_x
        near_y = self.y1 + t * dir_y
        return math.hypot(x - near_x, y - near_y) < self.width

    def in_path(self, ax, ay, tx, ty):
        barrier_dx = self.x2 - self.x1
        barrier_dy = self.y2 - self.y1
        path_dx = tx - ax
        path_dy = ty - ay
        cross = barrier_dx * path_dy - barrier_dy * path_dx
        return abs(cross) > 1e-6


class Agent:
    """Simple agent with position and heading."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0

    def step(self, thrust, turn, barrier=None):
        old_x, old_y = self.x, self.y

        self.angle += float(turn) * TURN_SCALE
        if thrust > 0.0:
            self.x += AGENT_SPEED * math.cos(self.angle)
            self.y += AGENT_SPEED * math.sin(self.angle)

        self.x = max(0.0, min(ARENA_SIZE - 1.0, self.x))
        self.y = max(0.0, min(ARENA_SIZE - 1.0, self.y))

        if barrier is not None and barrier.check_collision(self.x, self.y):
            self.x, self.y = old_x, old_y


def build_inputs(agent, target_x, target_y, barrier, expected_inputs):
    dx = target_x - agent.x
    dy = target_y - agent.y

    distance = math.hypot(dx, dy) / ARENA_SIZE
    angle_to_target = math.atan2(dy, dx) - agent.angle
    # Keep angle in [-pi, pi], then scale to [-1, 1].
    while angle_to_target > math.pi:
        angle_to_target -= 2.0 * math.pi
    while angle_to_target < -math.pi:
        angle_to_target += 2.0 * math.pi
    angle_norm = angle_to_target / math.pi

    inputs = [distance, angle_norm]
    if expected_inputs >= 3:
        in_way = 1.0 if (barrier is not None and barrier.in_path(agent.x, agent.y, target_x, target_y)) else 0.0
        inputs.append(in_way)
    while len(inputs) < expected_inputs:
        inputs.append(0.0)
    return inputs[:expected_inputs]


def run_episode(net, expected_inputs, barrier_length=0, rng=None):
    if rng is None:
        rng = random

    start_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
    start_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
    target_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
    target_y = rng.uniform(0.0, ARENA_SIZE - 1.0)

    agent = Agent(start_x, start_y)
    barrier = None
    if barrier_length and barrier_length > 0:
        barrier = Barrier(agent.x, agent.y, target_x, target_y, barrier_length)

    initial_distance = math.hypot(target_x - agent.x, target_y - agent.y)
    trajectory = [(agent.x, agent.y)]

    reached = False
    step = 0
    for step in range(MAX_STEPS):
        inputs = build_inputs(agent, target_x, target_y, barrier, expected_inputs)
        outputs = net.activate(inputs)

        thrust = outputs[0] if len(outputs) > 0 else 0.0
        turn = outputs[1] if len(outputs) > 1 else 0.0
        agent.step(thrust, turn, barrier)
        trajectory.append((agent.x, agent.y))

        if math.hypot(agent.x - target_x, agent.y - target_y) < TARGET_RADIUS:
            reached = True
            break

    final_distance = math.hypot(agent.x - target_x, agent.y - target_y)
    progress = max(0.0, initial_distance - final_distance)

    if reached:
        fitness = 1000.0 - 0.5 * step
    else:
        fitness = progress * 2.0 + max(0.0, 100.0 - 0.2 * final_distance)

    return {
        "fitness": fitness,
        "trajectory": trajectory,
        "start": (start_x, start_y),
        "target": (target_x, target_y),
        "barrier": barrier,
        "reached": reached,
        "steps": step + 1,
        "final_distance": final_distance,
    }


def save_trajectory_plot(path, episode):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.0, ARENA_SIZE)
    ax.set_ylim(0.0, ARENA_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    xs = [p[0] for p in episode["trajectory"]]
    ys = [p[1] for p in episode["trajectory"]]
    ax.plot(xs, ys, color="tab:red", linewidth=1.8, label="trajectory")

    sx, sy = episode["start"]
    tx, ty = episode["target"]
    ax.scatter([sx], [sy], c="tab:blue", s=40, label="start")
    ax.scatter([tx], [ty], c="tab:green", s=60, label="target")

    barrier = episode["barrier"]
    if barrier is not None:
        ax.plot(
            [barrier.x1, barrier.x2],
            [barrier.y1, barrier.y2],
            color="black",
            linewidth=3.0,
            label="barrier",
        )

    status = "reached" if episode["reached"] else "not reached"
    ax.set_title(
        f"Navigation rollout ({status}) | steps={episode['steps']} | "
        f"distance={episode['final_distance']:.2f}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def eval_genome(genome, config, barrier_length=0):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    expected_inputs = len(config.genome_config.input_keys)

    scores = []
    for _ in range(EVAL_EPISODES):
        result = run_episode(net, expected_inputs, barrier_length=barrier_length)
        scores.append(result["fitness"])
    return sum(scores) / len(scores)


def eval_genomes(genomes, config, barrier_length=0):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, barrier_length=barrier_length)


def get_node_names(config):
    node_names = {-1: "distance", -2: "angle_to_target"}
    if len(config.genome_config.input_keys) >= 3:
        node_names[-3] = "barrier_in_path"
    node_names[0] = "thrust"
    node_names[1] = "turn"
    return node_names


def save_run_artifacts(output_dir, config, genome, stats, node_names, barrier_length, view):
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

    # Save a deterministic rollout snapshot image for quick policy inspection.
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    expected_inputs = len(config.genome_config.input_keys)
    rng = random.Random(12345)
    episode = run_episode(net, expected_inputs, barrier_length=barrier_length, rng=rng)
    save_trajectory_plot(os.path.join(output_dir, "winner-trajectory.png"), episode)


class SnapshotReporter(neat.reporting.BaseReporter):
    def __init__(self, snapshot_interval, config, stats, node_names, barrier_length):
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.config = config
        self.stats = stats
        self.node_names = node_names
        self.barrier_length = barrier_length
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
            self.barrier_length,
            view=False,
        )
        print("\n" + "=" * 72)
        print(
            f" SNAPSHOT SAVED: generation {completed_generation:05d} -> "
            f"{os.path.abspath(snapshot_dir)}"
        )
        print("=" * 72 + "\n")


def run(config_file, barrier_length=0, generations=400):
    local_dir = os.path.dirname(__file__)
    config_basename = os.path.basename(config_file)
    run_dir = os.path.join(local_dir, f"exp-{config_basename}")
    os.makedirs(run_dir, exist_ok=True)

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
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        node_names = get_node_names(config)
        snapshot_interval = getattr(config, "snapshot_interval", 100)
        p.add_reporter(
            SnapshotReporter(
                snapshot_interval,
                config,
                stats,
                node_names,
                barrier_length,
            )
        )
        p.add_reporter(neat.Checkpointer(10))

        eval_fn = partial(eval_genome, barrier_length=barrier_length)
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_fn)
        winner = p.run(pe.evaluate, generations)

        print(f"\nRun directory: {run_dir}")
        print(f"\nBest genome:\n{winner!s}")
        save_run_artifacts(
            ".",
            config,
            winner,
            stats,
            node_names,
            barrier_length,
            view=True,
        )
    finally:
        os.chdir(previous_cwd)

    return winner, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run navigation feed-forward evolution with a chosen config file.",
    )
    parser.add_argument(
        "config_filename",
        nargs="?",
        default="config-feedforward",
        help="Config file name relative to this script, or an absolute path.",
    )
    parser.add_argument(
        "--barrier-length",
        type=float,
        default=0.0,
        help="Optional barrier length; 0 disables barriers.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=400,
        help="Maximum number of generations to run.",
    )
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if os.path.isabs(args.config_filename):
        config_path = args.config_filename
    else:
        config_path = os.path.join(local_dir, args.config_filename)

    run(
        config_path,
        barrier_length=args.barrier_length,
        generations=args.generations,
    )
