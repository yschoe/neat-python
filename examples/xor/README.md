# XOR Examples

Collection of XOR tasks showing minimal, feed-forward, spiking, parallel, and reproducible workflows.

## Evolve

Main feed-forward:
```bash
cd examples/xor
python evolve-feedforward.py [config_filename]
```

Other variants:
```bash
python evolve-minimal.py [config_filename]
python evolve-feedforward-partial.py [config_filename]
python evolve-feedforward-parallel.py [config_filename]
python evolve-feedforward-reproducible.py [config_filename]
python evolve-spiking.py [config_filename]
```

Configs include:
- `config-feedforward`, `config-feedforward-anji`
- `config-feedforward-partial`, `config-feedforward-partial-anji`
- `config-spiking`, `config-spiking-anji`

## Test

No separate `test-*.py`; scripts print winner outputs on XOR truth table.

## Task-specific config knobs

- tiny topology (`num_inputs=2`, `num_outputs=1`)
- mutation/reproduction rates for quick convergence
- `snapshot_interval` for periodic artifact saves

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner pickle
- fitness/speciation plots
- network dot/svg files
- snapshots
