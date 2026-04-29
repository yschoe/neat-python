# Single Pole Balancing

Evolves controllers for classic cart-pole balancing with feed-forward or CTRNN networks.

## Evolve

Feed-forward:
```bash
cd examples/single-pole-balancing
python evolve-feedforward.py [config_filename]
```

CTRNN:
```bash
python evolve-ctrnn.py [config_filename]
```

Common configs:
- `config-feedforward`
- `config-feedforward-anji`
- `config-ctrnn`
- `config-ctrnn-anji`

## Test

Feed-forward:
```bash
python test-feedforward.py [config_filename] [--snapshot N]
```

CTRNN:
```bash
python test-ctrnn.py [config_filename]
```

## Task-specific config knobs

- simulation horizon and balancing thresholds (script-level)
- network type (`feed_forward` vs recurrent/CTRNN)
- `fitness_threshold`, `pop_size`, `snapshot_interval`
- ANJI knobs in `*-anji` configs

## Outputs

Runtime dir: `exp-<config-filename>/`
- winner genome file
- fitness/speciation plots
- network dot/svg files
- optional movie output from test scripts
- `snapshot-xxxxx/`
