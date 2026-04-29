# Bipedal Walker (Feed-Forward)

Evolves a feed-forward controller for Gymnasium `BipedalWalker-v3`.

## Evolve

```bash
cd examples/bipedal-walker
python evolve-feedforward.py [config_filename]
```

Default config: `config-feedforward`.
ANJI mode config: `config-feedforward-anji`.

## Test

```bash
python test-feedforward.py [config_filename] [genome_path] [--snapshot N]
```

- `config_filename`: optional (default `config-feedforward`)
- `genome_path`: optional explicit winner pickle
- `--snapshot N`: load from `snapshot-<N padded to 5 digits>`

## Task-specific config knobs

- `num_inputs=24`, `num_outputs=4`
- `fitness_threshold`: target reward to stop evolution
- `pop_size`: evolutionary population size
- `snapshot_interval` (`[NEAT]`): periodic artifact save interval

## Outputs

Runtime dir: `exp-<config-filename>/`

Generated artifacts include:
- `winner-feedforward.pickle`
- `feedforward-fitness.svg`
- `feedforward-speciation.svg`
- `winner-feedforward.gv(.svg)`
- `winner-feedforward-pruned.gv(.svg)`
- `snapshot-xxxxx/` (periodic snapshots)
