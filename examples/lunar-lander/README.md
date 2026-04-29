# Lunar Lander (Feed-Forward)

Evolves a feed-forward controller for Gymnasium `LunarLander-v3`.

## Evolve

```bash
cd examples/lunar-lander
python evolve-feedforward.py [config_filename]
```

Common configs:
- `config-feedforward`
- `config-feedforward-anji*`

## Test

```bash
python test-feedforward.py [config_filename] [genome_path] [--snapshot N]
```

## Task-specific config knobs

- `num_inputs=8`, `num_outputs=4` (discrete action chosen by argmax output)
- `fitness_threshold` (typically around solved-score target)
- `pop_size`
- `snapshot_interval`
- ANJI knobs: `algorithm_mode=anji`, `anji_prune`, `anji_remove_connection_rate`, etc.

## Outputs

Runtime dir: `exp-<config-filename>/`
- `winner-feedforward.pickle`
- `feedforward-fitness.svg`
- `feedforward-speciation.svg`
- `winner-feedforward.gv(.svg)`
- `winner-feedforward-pruned.gv(.svg)`
- `snapshot-xxxxx/`
