"""\
Test and visualize the best genome produced by evolve-feedforward.py.
"""

import argparse
import math
import os
import pickle
import random
import time

import neat


ARENA_SIZE = 300
MAX_STEPS = 600
TARGET_RADIUS = 10.0
AGENT_SPEED = 3.0
TURN_SCALE = 0.2


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


class Barrier:
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


def build_inputs(agent, target_x, target_y, barrier, expected_inputs, barrier_sensor_enabled=True):
    dx = target_x - agent.x
    dy = target_y - agent.y
    distance = math.hypot(dx, dy) / ARENA_SIZE

    angle = math.atan2(dy, dx) - agent.angle
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    angle_norm = angle / math.pi

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


def _latest_values(net):
    # FeedForwardNetwork stores one dict; RecurrentNetwork stores two alternating dicts.
    if hasattr(net, "active") and hasattr(net, "values") and isinstance(net.values, list):
        return net.values[net.active]
    if hasattr(net, "values") and isinstance(net.values, dict):
        return net.values
    return {}


def _activity_rows(config, genome, net, current_inputs):
    values = _latest_values(net)
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    hidden_keys = sorted(
        [k for k in genome.nodes.keys() if k not in set(output_keys)]
    )

    input_vals = {
        k: (current_inputs[idx] if idx < len(current_inputs) else 0.0)
        for idx, k in enumerate(input_keys)
    }
    hidden_vals = {k: values.get(k, 0.0) for k in hidden_keys}
    output_vals = {k: values.get(k, 0.0) for k in output_keys}
    return output_vals, hidden_vals, input_vals


def _value_color(v):
    # Map roughly [-1, 1] to red/blue with white near zero.
    x = max(-1.0, min(1.0, float(v)))
    if x >= 0:
        r = 255
        g = int(255 * (1.0 - x))
        b = int(255 * (1.0 - x))
    else:
        r = int(255 * (1.0 + x))
        g = int(255 * (1.0 + x))
        b = 255
    return (r, g, b)


def _draw_activity_panel(surface, rows):
    import pygame

    width, height = surface.get_size()
    surface.fill((22, 22, 22))
    row_names = ["outputs", "hidden", "inputs"]
    y_positions = [height * 0.18, height * 0.5, height * 0.82]

    try:
        font = pygame.font.SysFont("monospace", 14)
    except Exception:
        font = None

    for row_name, row_vals, y in zip(row_names, rows, y_positions):
        keys = list(row_vals.keys())
        n = max(1, len(keys))
        for i, key in enumerate(keys):
            x = int((i + 1) * width / (n + 1))
            v = row_vals[key]
            color = _value_color(v)
            pygame.draw.circle(surface, color, (x, int(y)), 13)
            pygame.draw.circle(surface, (240, 240, 240), (x, int(y)), 13, 1)
            if font is not None:
                label = font.render(str(key), True, (220, 220, 220))
                valtxt = font.render(f"{v:+.2f}", True, (200, 200, 200))
                surface.blit(label, (x - label.get_width() // 2, int(y) + 18))
                surface.blit(valtxt, (x - valtxt.get_width() // 2, int(y) - 30))

        if font is not None:
            title = font.render(row_name, True, (180, 180, 180))
            surface.blit(title, (8, int(y) - 8))


def draw_episode(frame_path, episode):
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
    ax.scatter([episode["start"][0]], [episode["start"][1]], c="tab:blue", s=40, label="start")
    ax.scatter([episode["target"][0]], [episode["target"][1]], c="tab:green", s=60, label="target")

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
    fig.savefig(frame_path)
    plt.close(fig)


def run_episode(
    net,
    expected_inputs,
    barrier_length=0.0,
    seed=None,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
    rng = random.Random(seed)
    min_start_target_distance = ARENA_SIZE / 3.0
    for _ in range(200):
        start_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        start_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        if math.hypot(target_x - start_x, target_y - start_y) >= min_start_target_distance:
            break
    else:
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

    trajectory = [(agent.x, agent.y)]
    step = 0
    reached = False
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
    return {
        "start": (start_x, start_y),
        "target": (target_x, target_y),
        "barrier": barrier,
        "trajectory": trajectory,
        "reached": reached,
        "steps": step + 1,
        "final_distance": final_distance,
    }


def render_episode(episode):
    try:
        import pygame
    except ImportError:
        print("pygame is not installed; skipping interactive render.")
        return

    pygame.init()
    screen = pygame.display.set_mode((ARENA_SIZE, ARENA_SIZE))
    pygame.display.set_caption("Navigation test rollout")
    clock = pygame.time.Clock()

    trajectory = episode["trajectory"]
    barrier = episode["barrier"]
    target = episode["target"]
    finished = False
    idx = 0
    hold_until = None

    while not finished:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True

        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (0, 220, 0), (int(target[0]), int(target[1])), int(TARGET_RADIUS))

        if barrier is not None:
            pygame.draw.line(
                screen,
                (255, 255, 255),
                (int(barrier.x1), int(barrier.y1)),
                (int(barrier.x2), int(barrier.y2)),
                int(max(1.0, barrier.width)),
            )

        if idx > 1:
            pygame.draw.lines(
                screen,
                (100, 100, 255),
                False,
                [(int(p[0]), int(p[1])) for p in trajectory[: idx + 1]],
                2,
            )

        x, y = trajectory[idx]
        pygame.draw.circle(screen, (255, 60, 60), (int(x), int(y)), 7)

        pygame.display.flip()
        clock.tick(60)

        if idx < len(trajectory) - 1:
            idx += 1
        else:
            if hold_until is None:
                hold_until = time.time() + 2.0
            elif time.time() >= hold_until:
                finished = True

    pygame.quit()


def render_episode_with_activity(
    net,
    config,
    genome,
    expected_inputs,
    barrier_length=0.0,
    seed=None,
    barrier_sensor_enabled=True,
    angled_barrier=False,
):
    try:
        import pygame
    except ImportError:
        print("pygame is not installed; skipping interactive render.")
        return run_episode(
            net,
            expected_inputs,
            barrier_length=barrier_length,
            seed=seed,
            barrier_sensor_enabled=barrier_sensor_enabled,
            angled_barrier=angled_barrier,
        )

    rng = random.Random(seed)
    min_start_target_distance = ARENA_SIZE / 3.0
    for _ in range(200):
        start_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        start_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_x = rng.uniform(0.0, ARENA_SIZE - 1.0)
        target_y = rng.uniform(0.0, ARENA_SIZE - 1.0)
        if math.hypot(target_x - start_x, target_y - start_y) >= min_start_target_distance:
            break
    else:
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

    pygame.init()
    panel_width = 360
    screen = pygame.display.set_mode((ARENA_SIZE + panel_width, ARENA_SIZE))
    pygame.display.set_caption("Navigation rollout + neural activity")
    clock = pygame.time.Clock()
    arena_surface = pygame.Surface((ARENA_SIZE, ARENA_SIZE))
    panel_surface = pygame.Surface((panel_width, ARENA_SIZE))

    trajectory = [(agent.x, agent.y)]
    reached = False
    step = 0
    running = True
    hold_until = None
    current_inputs = [0.0] * expected_inputs

    while running and step < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_inputs = build_inputs(
            agent,
            target_x,
            target_y,
            barrier,
            expected_inputs,
            barrier_sensor_enabled=barrier_sensor_enabled,
        )
        outputs = net.activate(current_inputs)
        thrust = outputs[0] if len(outputs) > 0 else 0.0
        turn = outputs[1] if len(outputs) > 1 else 0.0
        agent.step(thrust, turn, barrier)
        trajectory.append((agent.x, agent.y))
        step += 1

        if math.hypot(agent.x - target_x, agent.y - target_y) < TARGET_RADIUS:
            reached = True
            if hold_until is None:
                hold_until = time.time() + 2.0

        arena_surface.fill((0, 0, 0))
        pygame.draw.circle(arena_surface, (0, 220, 0), (int(target_x), int(target_y)), int(TARGET_RADIUS))
        if barrier is not None:
            for s1, s2 in barrier.segments:
                pygame.draw.line(
                    arena_surface,
                    (255, 255, 255),
                    (int(s1[0]), int(s1[1])),
                    (int(s2[0]), int(s2[1])),
                    int(max(1.0, barrier.width)),
                )
        if len(trajectory) > 1:
            pygame.draw.lines(
                arena_surface,
                (100, 100, 255),
                False,
                [(int(p[0]), int(p[1])) for p in trajectory],
                2,
            )
        pygame.draw.circle(arena_surface, (255, 60, 60), (int(agent.x), int(agent.y)), 7)

        out_vals, hid_vals, in_vals = _activity_rows(config, genome, net, current_inputs)
        _draw_activity_panel(panel_surface, [out_vals, hid_vals, in_vals])

        screen.blit(arena_surface, (0, 0))
        screen.blit(panel_surface, (ARENA_SIZE, 0))
        pygame.display.flip()
        clock.tick(60)

        if hold_until is not None and time.time() >= hold_until:
            break

    pygame.quit()

    final_distance = math.hypot(agent.x - target_x, agent.y - target_y)
    return {
        "start": (start_x, start_y),
        "target": (target_x, target_y),
        "barrier": barrier,
        "trajectory": trajectory,
        "reached": reached,
        "steps": step,
        "final_distance": final_distance,
    }


def load_and_test(
    genome_path,
    config_path,
    barrier_length=0.0,
    episodes=10,
    render=True,
    disable_barrier_sensor=False,
    angled_barrier=False,
):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print("Loaded genome:")
    print(genome)

    barrier_length = resolve_barrier_length(config, barrier_length)
    barrier_sensor_enabled = resolve_barrier_sensor_enabled(config, disable_barrier_sensor)
    angled_barrier = resolve_angled_barrier_enabled(config, angled_barrier)
    net = create_network(genome, config)
    expected_inputs = len(config.genome_config.input_keys)

    for episode_idx in range(episodes):
        if render:
            episode = render_episode_with_activity(
                net,
                config,
                genome,
                expected_inputs,
                barrier_length=barrier_length,
                seed=episode_idx + 1,
                barrier_sensor_enabled=barrier_sensor_enabled,
                angled_barrier=angled_barrier,
            )
        else:
            episode = run_episode(
                net,
                expected_inputs,
                barrier_length=barrier_length,
                seed=episode_idx + 1,
                barrier_sensor_enabled=barrier_sensor_enabled,
                angled_barrier=angled_barrier,
            )
        status = "reached" if episode["reached"] else "not reached"
        print(
            f"Episode {episode_idx + 1}: {status}, steps={episode['steps']}, "
            f"final_distance={episode['final_distance']:.2f}"
        )

        frame_name = f"test-episode-{episode_idx + 1:02d}.png"
        draw_episode(frame_name, episode)
        print(f"Saved trajectory image: {os.path.abspath(frame_name)}")



def resolve_run_dir_and_genome(local_dir, config_path, genome_path=None, snapshot=None):
    if genome_path is not None:
        return os.path.dirname(os.path.abspath(genome_path)) or os.getcwd(), genome_path

    winner_name = "winner-feedforward.pickle"
    config_basename = os.path.basename(config_path)

    if snapshot is not None:
        run_dir = os.path.join(
            local_dir,
            f"exp-{config_basename}",
            f"snapshot-{snapshot:05d}",
        )
        return run_dir, os.path.join(run_dir, winner_name)

    cwd_genome = os.path.join(os.getcwd(), winner_name)
    if os.path.exists(cwd_genome):
        return os.getcwd(), cwd_genome

    run_dir = os.path.join(local_dir, f"exp-{config_basename}")
    return run_dir, os.path.join(run_dir, winner_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test navigation winner with a chosen config file.",
    )
    parser.add_argument(
        "config_filename",
        nargs="?",
        default="config-feedforward",
        help="Config file name relative to this script, or an absolute path.",
    )
    parser.add_argument(
        "genome_path",
        nargs="?",
        default=None,
        help="Optional explicit path to winner genome.",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help="Snapshot generation number to load (e.g. 100 loads snapshot-00100).",
    )
    parser.add_argument(
        "--barrier-length",
        type=float,
        default=None,
        help="Optional barrier length override; if omitted, read from config (if present).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of rollout episodes to run.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable interactive pygame rendering.",
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

    run_dir, genome_path = resolve_run_dir_and_genome(
        local_dir,
        config_path,
        args.genome_path,
        snapshot=args.snapshot,
    )

    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}")
        print(f"Checked cwd and run directory: {run_dir}")
        print("Please train a network first by running evolve-feedforward.py")
        raise SystemExit(1)

    print(f"Run directory: {run_dir}")
    print(f"Testing genome from: {genome_path}\n")

    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        load_and_test(
            genome_path,
            config_path,
            barrier_length=args.barrier_length,
            episodes=max(1, args.episodes),
            render=not args.no_render,
            disable_barrier_sensor=args.disable_barrier_sensor,
            angled_barrier=args.angled_barrier,
        )
    finally:
        os.chdir(previous_cwd)
