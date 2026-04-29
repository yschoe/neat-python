# Inverted Double Pendulum (Feed-Forward)

Evolves a feed-forward controller for Gymnasium `InvertedDoublePendulum-v5` (MuJoCo).

## Evolve

```bash
cd examples/inverted-double-pendulum
python evolve-feedforward.py [config_filename]
```

Common configs:
- `config-feedforward`
- `config-feedforward-anji` (and variants)

## Test

```bash
python test-feedforward.py [config_filename] [genome_path] [--snapshot N]
```

## Task-specific config knobs

- `num_inputs=9`, `num_outputs=1`
- `fitness_threshold`
- `pop_size`
- `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle, fitness/speciation plots, network dot/svg, snapshots
