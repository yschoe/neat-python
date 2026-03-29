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

## 2026-03-29

- Updated all `examples/*/evolve*.py` scripts to:
  - accept optional config filename arguments
  - write outputs into `exp-<config-filename>/` runtime directories
- Updated all `examples/*/test-*.py` scripts to:
  - accept optional config filename arguments
  - resolve winner files from current working directory first, then `exp-<config-filename>/`
- Added explicit ANJI compatibility config keys to all `examples/*/config*anji*` files:
  - genome keys: `anji_topology_mutation_classic`, `anji_mutate_nodes`, `anji_prune`, `anji_remove_connection_rate`
  - reproduction keys: `anji_survival_rate`, `anji_elitism`, `anji_elitism_min_species_size`, `anji_clone_slice`, `anji_crossover_slice`
- Committed and pushed:
  - `9742452` (`Examples: add config-driven exp dirs and explicit ANJI knobs`) to `master`

- Updated network topology plotting layout in all example `visualize.py` modules:
  - set `rankdir=BT`
  - force input nodes to `rank=min`
  - force output nodes to `rank=max`
- Added DOT converter utility:
  - `tools/fixdot.py`
  - converts old `.gv/.dot` to BT-ranked layout and renders `.svg`
