# Memory Fixed-Length Sequence

Experimental task: evolve a network to reproduce a fixed-length binary sequence.

## Evolve

```bash
cd examples/memory-fixed
python evolve.py [config_filename]
```

Configs:
- `config`
- `config-anji`

## Test

No separate `test-*.py`; inspect fitness and winner behavior from run artifacts.

## Task-specific config knobs

- sequence/task parameters embedded in script
- network type and recurrence options in config
- `fitness_threshold`, `pop_size`, `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle
- fitness/speciation plots
- network dot/svg
- snapshots
