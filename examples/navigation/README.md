# Navigation example

This example evolves a controller for a 2D navigation task.
An agent moves from random start to random target in a square arena.

> Note: this example is still in an experimental stage and may change.

## Evolve

```bash
cd examples/navigation
python evolve-feedforward.py [config_filename] [--barrier-length L] [--generations N] [--step-penalty-success P] [--step-penalty-all Q]
```

Configs:
- `config-feedforward`
- `config-feedforward-anji`
- `config-rnn`
- `config-rnn-anji`

The script chooses network type from config:
- `feed_forward = True` -> `FeedForwardNetwork`
- `feed_forward = False` -> `RecurrentNetwork`

## Test

```bash
python test-feedforward.py [config_filename] [genome_path] [--snapshot N] [--barrier-length L] [--episodes N] [--no-render]
```

When rendering is enabled (default), the test window now includes a real-time
neuron activity panel:
- top row: output neurons
- middle row: hidden neurons
- bottom row: input neurons

No connectivity is drawn; node color/value indicates activation level.

## Task-specific config knobs

- `num_inputs=3` (`distance`, `angle`, `barrier_in_path`)
  - `barrier_in_path` is a near-contact binary sensor (1 when within ~5 px of barrier, else 0)
- `num_outputs=2` (`thrust`, `turn`)
- `feed_forward` controls network type (`True` feed-forward, `False` recurrent)
- `snapshot_interval`
- ANJI mode knobs in `config-feedforward-anji`
- `barrier_length` and `barrier_in_path` can be set in config;
  `--barrier-length` overrides config at runtime.
- Fitness step penalties (CLI):
  - `--step-penalty-success P` (default `0.5`) applies when target is reached
  - `--step-penalty-all Q` (default `0.0`) applies to all episodes

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle
- fitness/speciation plots
- network dot/svg
- `winner-trajectory.png`
- `test-episode-*.png` from testing
- `snapshot-xxxxx/`
