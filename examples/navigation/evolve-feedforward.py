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
STEP_PENALTY_SUCCESS = 0.5
STEP_PENALTY_ALL = 0.0


class Barrier:
    """Line-segment barrier placed between start and target."""

    def __init__(
        self,
        start_x,
        start_y,
        target_x,
        target_y,
        length=100.0,
        rng=None,
        angled_barrier=False,
    ):
        if rng is None:
            rng = random

        dx = target_x - start_x
        dy = target_y - start_y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            unit_dx, unit_dy = 1.0, 0.0
            perp_dx, perp_dy = 0.0, 1.0
            path_fraction = 0.5
        else:
            unit_dx, unit_dy = dx / norm, dy / norm
            perp_dx, perp_dy = -dy / norm, dx / norm
            # Place barrier between 1/4 and 3/4 along agent->target path.
            path_fraction = rng.uniform(0.25, 0.75)

        mid_x = start_x + unit_dx * (norm * path_fraction)
        mid_y = start_y + unit_dy * (norm * path_fraction)

        half = 0.5 * float(length)
        self.x1 = mid_x + perp_dx * half
        self.y1 = mid_y + perp_dy * half
        self.x2 = mid_x - perp_dx * half
        self.y2 = mid_y - perp_dy * half
        self.width = 5.0
        self.angled_barrier = bool(angled_barrier)
        self.endcap_length = max(1.0, 0.2 * float(length))
        self.start_x = start_x
        self.start_y = start_y
        self.segments = self._build_segments()

    def _build_segments(self):
        segments = [((self.x1, self.y1), (self.x2, self.y2))]
        if not self.angled_barrier:
            return segments

        for ex, ey, other_x, other_y in (
            (self.x1, self.y1, self.x2, self.y2),
            (self.x2, self.y2, self.x1, self.y1),
        ):
            tx = other_x - ex
            ty = other_y - ey
            tnorm = math.hypot(tx, ty)
            if tnorm < 1e-9:
                continue
            tx /= tnorm
            ty /= tnorm

            ax = self.start_x - ex
            ay = self.start_y - ey
            anorm = math.hypot(ax, ay)
            if anorm < 1e-9:
                ax, ay = -tx, -ty
                anorm = 1.0
            ax /= anorm
            ay /= anorm

            # Strict 90-degree endcap from the main barrier tangent.
            # Pick the perpendicular that points toward the agent side.
            nx1, ny1 = -ty, tx
            nx2, ny2 = ty, -tx
            dot1 = nx1 * ax + ny1 * ay
            dot2 = nx2 * ax + ny2 * ay
            if dot1 >= dot2:
                dx, dy = nx1, ny1
            else:
                dx, dy = nx2, ny2

            cap_x = ex + dx * self.endcap_length
            cap_y = ey + dy * self.endcap_length
            segments.append(((ex, ey), (cap_x, cap_y)))

        return segments

    @staticmethod
    def _distance_point_to_segment(x, y, sx1, sy1, sx2, sy2):
        seg_len = math.hypot(sx2 - sx1, sy2 - sy1)
        if seg_len < 1e-9:
            return 1e9

        dir_x = (sx2 - sx1) / seg_len
        dir_y = (sy2 - sy1) / seg_len
        rel_x = x - sx1
        rel_y = y - sy1
        t = max(0.0, min(seg_len, dir_x * rel_x + dir_y * rel_y))
        near_x = sx1 + t * dir_x
        near_y = sy1 + t * dir_y
        return math.hypot(x - near_x, y - near_y)

    def check_collision(self, x, y):
        return self.distance_to_point(x, y) < self.width

    def distance_to_point(self, x, y):
        return min(
            self._distance_point_to_segment(x, y, s1[0], s1[1], s2[0], s2[1])
            for s1, s2 in self.segments
        )


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


def create_network(genome, config):
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return neat.nn.RecurrentNetwork.create(genome, config)


def resolve_barrier_length(config, cli_barrier_length):
    if cli_barrier_length is not None:
        return max(0.0, float(cli_barrier_length))

    cfg_len = float(getattr(config.genome_config, "barrier_length", 0.0))
    use_barrier = str(getattr(config.genome_config, "barrier_in_path", "true")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return max(0.0, cfg_len) if use_barrier else 0.0


def resolve_barrier_sensor_enabled(config, cli_disable_barrier_sensor):
    if cli_disable_barrier_sensor:
        return False
    return str(getattr(config.genome_config, "barrier_sensor_enabled", "true")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_angled_barrier_enabled(config, cli_angled_barrier):
    if cli_angled_barrier:
        return True
    return str(getattr(config.genome_config, "angled_barrier", "false")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def build_inputs(agent, target_x, target_y, barrier, expected_inputs, barrier_sensor_enabled=True):
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
        # Barrier proximity sensor with short lookahead so it fires before a
        # move that would collide and be reverted by physics.
        near_barrier = 0.0
        if barrier_sensor_enabled:
            near_now = False
            near_next = False
            if barrier is not None:
                near_now = barrier.distance_to_point(agent.x, agent.y) <= 5.0
                probe_x = agent.x + AGENT_SPEED * math.cos(agent.angle)
                probe_y = agent.y + AGENT_SPEED * math.sin(agent.angle)
                near_next = barrier.distance_to_point(probe_x, probe_y) <= (barrier.width + 1.0)
            near_barrier = 1.0 if (near_now or near_next) else 0.0
        inputs.append(near_barrier)
    while len(inputs) < expected_inputs:
        inputs.append(0.0)
    return inputs[:expected_inputs]


def run_episode(
    net,
    expected_inputs,
    barrier_length=0,
    rng=None,
    step_penalty_success=STEP_PENALTY_SUCCESS,
    step_penalty_all=STEP_PENALTY_ALL,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
    if rng is None:
        rng = random

    min_start_target_distance = ARENA_SIZE / 3.0
    for _ in range(200):
        start_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        start_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        if math.hypot(target_x - start_x, target_y - start_y) >= min_start_target_distance:
            break
    else:
        # Fallback if random sampling fails repeatedly.
        start_x, start_y = 0.0, 0.0
        target_x, target_y = ARENA_SIZE - 1.0, ARENA_SIZE - 1.0

    agent = Agent(start_x, start_y)
    if hasattr(net, "reset"):
        net.reset()
    barrier = None
    if barrier_length and barrier_length > 0:
        barrier = Barrier(
            agent.x,
            agent.y,
            target_x,
            target_y,
            barrier_length,
            rng=rng,
            angled_barrier=angled_barrier,
        )

    initial_distance = math.hypot(target_x - agent.x, target_y - agent.y)
    trajectory = [(agent.x, agent.y)]

    reached = False
    step = 0
    for step in range(MAX_STEPS):
        inputs = build_inputs(
            agent,
            target_x,
            target_y,
            barrier,
            expected_inputs,
            barrier_sensor_enabled=barrier_sensor_enabled,
        )
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
        fitness = 1000.0 - float(step_penalty_success) * step
    else:
        fitness = progress * 2.0 + max(0.0, 100.0 - 0.2 * final_distance)
    fitness -= float(step_penalty_all) * step

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
        for idx, (s1, s2) in enumerate(barrier.segments):
            ax.plot(
                [s1[0], s2[0]],
                [s1[1], s2[1]],
                color="black",
                linewidth=3.0,
                label="barrier" if idx == 0 else None,
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


def eval_genome(
    genome,
    config,
    barrier_length=0,
    step_penalty_success=STEP_PENALTY_SUCCESS,
    step_penalty_all=STEP_PENALTY_ALL,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
    net = create_network(genome, config)
    expected_inputs = len(config.genome_config.input_keys)

    scores = []
    for _ in range(EVAL_EPISODES):
        result = run_episode(
            net,
            expected_inputs,
            barrier_length=barrier_length,
            step_penalty_success=step_penalty_success,
            step_penalty_all=step_penalty_all,
            barrier_sensor_enabled=barrier_sensor_enabled,
            angled_barrier=angled_barrier,
        )
        scores.append(result["fitness"])
    return sum(scores) / len(scores)


def eval_genomes(
    genomes,
    config,
    barrier_length=0,
    step_penalty_success=STEP_PENALTY_SUCCESS,
    step_penalty_all=STEP_PENALTY_ALL,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(
            genome,
            config,
            barrier_length=barrier_length,
            step_penalty_success=step_penalty_success,
            step_penalty_all=step_penalty_all,
            barrier_sensor_enabled=barrier_sensor_enabled,
            angled_barrier=angled_barrier,
        )


def get_node_names(config):
    node_names = {-1: "distance", -2: "angle_to_target"}
    if len(config.genome_config.input_keys) >= 3:
        node_names[-3] = "barrier_in_path"
    node_names[0] = "thrust"
    node_names[1] = "turn"
    return node_names


def save_run_artifacts(
    output_dir,
    config,
    genome,
    stats,
    node_names,
    barrier_length,
    view,
    step_penalty_success=STEP_PENALTY_SUCCESS,
    step_penalty_all=STEP_PENALTY_ALL,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
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
    net = create_network(genome, config)
    expected_inputs = len(config.genome_config.input_keys)
    rng = random.Random(12345)
    episode = run_episode(
        net,
        expected_inputs,
        barrier_length=barrier_length,
        rng=rng,
        step_penalty_success=step_penalty_success,
        step_penalty_all=step_penalty_all,
        barrier_sensor_enabled=barrier_sensor_enabled,
        angled_barrier=angled_barrier,
    )
    save_trajectory_plot(os.path.join(output_dir, "winner-trajectory.png"), episode)


class SnapshotReporter(neat.reporting.BaseReporter):
    def __init__(
        self,
        snapshot_interval,
        config,
        stats,
        node_names,
        barrier_length,
        step_penalty_success,
        step_penalty_all,
        barrier_sensor_enabled,
        angled_barrier,
    ):
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.config = config
        self.stats = stats
        self.node_names = node_names
        self.barrier_length = barrier_length
        self.step_penalty_success = float(step_penalty_success)
        self.step_penalty_all = float(step_penalty_all)
        self.barrier_sensor_enabled = bool(barrier_sensor_enabled)
        self.angled_barrier = bool(angled_barrier)
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
            step_penalty_success=self.step_penalty_success,
            step_penalty_all=self.step_penalty_all,
            barrier_sensor_enabled=self.barrier_sensor_enabled,
            angled_barrier=self.angled_barrier,
        )
        print("\n" + "=" * 72)
        print(
            f" SNAPSHOT SAVED: generation {completed_generation:05d} -> "
            f"{os.path.abspath(snapshot_dir)}"
        )
        print("=" * 72 + "\n")


def run(
    config_file,
    barrier_length=None,
    generations=400,
    step_penalty_success=STEP_PENALTY_SUCCESS,
    step_penalty_all=STEP_PENALTY_ALL,
    disable_barrier_sensor=False,
    angled_barrier=False,
):
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
    barrier_length = resolve_barrier_length(config, barrier_length)
    barrier_sensor_enabled = resolve_barrier_sensor_enabled(config, disable_barrier_sensor)
    angled_barrier = resolve_angled_barrier_enabled(config, angled_barrier)

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
                step_penalty_success,
                step_penalty_all,
                barrier_sensor_enabled,
                angled_barrier,
            )
        )
        p.add_reporter(neat.Checkpointer(10))

        eval_fn = partial(
            eval_genome,
            barrier_length=barrier_length,
            step_penalty_success=step_penalty_success,
            step_penalty_all=step_penalty_all,
            barrier_sensor_enabled=barrier_sensor_enabled,
            angled_barrier=angled_barrier,
        )
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
            step_penalty_success=step_penalty_success,
            step_penalty_all=step_penalty_all,
            barrier_sensor_enabled=barrier_sensor_enabled,
            angled_barrier=angled_barrier,
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
        default=None,
        help="Optional barrier length override; if omitted, read from config (if present).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=400,
        help="Maximum number of generations to run.",
    )
    parser.add_argument(
        "--step-penalty-success",
        type=float,
        default=STEP_PENALTY_SUCCESS,
        help="Penalty multiplier on steps for successful episodes (default: 0.5).",
    )
    parser.add_argument(
        "--step-penalty-all",
        type=float,
        default=STEP_PENALTY_ALL,
        help="Additional penalty multiplier on steps for all episodes (default: 0.0).",
    )
    parser.add_argument(
        "--disable-barrier-sensor",
        action="store_true",
        help="Force-disable the barrier sensor input (third input becomes 0).",
    )
    parser.add_argument(
        "--angled-barrier",
        action="store_true",
        help="Enable short angled end segments at both barrier ends toward the agent side.",
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
        step_penalty_success=args.step_penalty_success,
        step_penalty_all=args.step_penalty_all,
        disable_barrier_sensor=args.disable_barrier_sensor,
        angled_barrier=args.angled_barrier,
    )
