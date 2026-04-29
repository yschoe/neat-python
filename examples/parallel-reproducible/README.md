# Parallel Reproducible Evolution

Demonstrates reproducible NEAT runs with parallel evaluation.

## Evolve

```bash
cd examples/parallel-reproducible
python evolve-parallel.py [config_filename]
```

Configs:
- `config-parallel`
- `config-parallel-anji`

## Test

No separate `test-*.py`; reproducibility is checked by repeated runs and comparing outputs.

## Task-specific config knobs

- worker/evaluation setup in script
- seed handling for deterministic parallel behavior
- `pop_size`, mutation rates, `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner and stats artifacts similar to other evolve examples
