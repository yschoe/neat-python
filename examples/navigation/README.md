# Navigation example

This example evolves a feed-forward controller for a 2D navigation task.
An agent moves from random start to random target in a square arena.

> Note: this example is still in an experimental stage and may change.

## Evolve

```bash
cd examples/navigation
python evolve-feedforward.py [config_filename] [--barrier-length L] [--generations N]
```

Configs:
- `config-feedforward`
- `config-feedforward-anji`

## Test

```bash
python test-feedforward.py [config_filename] [genome_path] [--snapshot N] [--barrier-length L] [--episodes N] [--no-render]
```

## Task-specific config knobs

- `num_inputs=3` (`distance`, `angle`, `barrier_in_path`)
- `num_outputs=2` (`thrust`, `turn`)
- `snapshot_interval`
- ANJI mode knobs in `config-feedforward-anji`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle
- fitness/speciation plots
- network dot/svg
- `winner-trajectory.png`
- `test-episode-*.png` from testing
- `snapshot-xxxxx/`
