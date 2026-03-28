# Log

## 2026-03-28

- Added ANJI compatibility mode to neat-python via `algorithm_mode = anji`.
- Added parity and comparison reports under `python-vs-java/`.
- Added single-pole-balancing ANJI config:
  - `examples/single-pole-balancing/config-feedforward-anji`
- Updated single-pole-balancing scripts:
  - `evolve-feedforward.py` now accepts config filename argument and writes outputs to `exp-<configfilename>/`.
  - `test-feedforward.py` now accepts config filename argument and reads/writes from `exp-<configfilename>/`.
- Added experiment output directories to version control:
  - `examples/single-pole-balancing/exp-config-feedforward/`
  - `examples/single-pole-balancing/exp-config-feedforward-anji/`
