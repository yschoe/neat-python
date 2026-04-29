# Memory Variable-Length Sequence

Experimental task: evolve a network to reproduce variable-length binary sequences.

## Evolve

```bash
cd examples/memory-variable
python evolve.py [config_filename]
```

Configs:
- `config`
- `config-anji`

## Test

No separate `test-*.py`; inspect run artifacts and winner behavior.

## Task-specific config knobs

- recurrence/network settings
- mutation/reproduction rates
- `fitness_threshold`, `pop_size`, `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle
- fitness/speciation plots
- network dot/svg
- snapshots
