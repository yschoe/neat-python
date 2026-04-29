# Spike Timing (GPU)

Evolves spiking-network timing behavior with optional GPU acceleration.

## Evolve

```bash
cd examples/spike-timing-gpu
python evolve.py [config_filename] [--cpu-only] [--gpu-only] [--generations N] [--pop-size N] [--seed N]
```

Configs:
- `config-iznn`
- `config-iznn-anji`

## Test

No separate `test-*.py`; evaluate with generated stats and winner behavior.

## Task-specific config knobs

- Izhikevich neuron parameters
- timing objective settings in script/config
- CPU/GPU mode flags
- `snapshot_interval`

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner genome
- plots and logs from evolution run
- snapshots
