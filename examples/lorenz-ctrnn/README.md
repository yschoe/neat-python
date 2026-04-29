# Lorenz CTRNN

Evolves CTRNNs for Lorenz-system signal prediction/tracking variants.

## Evolve

```bash
cd examples/lorenz-ctrnn
python evolve_lorenz_ctrnn.py [config_filename] [--mode {base,products,product-agg}] [--z-only] [--workers N]
```

Defaults:
- `config_filename`: `config-ctrnn`
- ANJI config: `config-ctrnn-anji`

## Test

No separate `test-*.py`; evaluate by rerunning and inspecting final metrics/plots.

## Task-specific config knobs

- CTRNN dynamics: `time_const_*`, activation/aggregation, recurrence settings
- `fitness_threshold`, `pop_size`, `max_stagnation`
- `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner genome pickle
- fitness/speciation plots
- network graph files
- snapshots (if interval reached)
