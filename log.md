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

- Added snapshot support across examples:
  - New optional `[NEAT]` config key: `snapshot_interval` (default `100` when omitted)
  - `evolve*.py` scripts now save periodic snapshots under:
    - `exp-<config-filename>/snapshot-<generation:05d>/`
  - Snapshot saves are non-blocking (`view=False`) and print prominent console messages.
  - Final-generation outputs in `exp-<config-filename>/` remain unchanged.
- Updated all `test-feedforward.py` scripts to support:
  - `--snapshot <generation>` to load `snapshot-<generation:05d>` winners.

## 2026-03-30

- Added new experimental example: `examples/navigation/`, ported from `scratch/neat-navi/`.
- Added standardized navigation example files:
  - `evolve-feedforward.py`
  - `test-feedforward.py`
  - `visualize.py`
  - `config-feedforward`
  - `config-feedforward-anji`
  - `README.md`
- Matched repository conventions:
  - optional config filename CLI argument
  - runtime output directory `exp-<config-filename>/`
  - snapshot saves via `snapshot_interval` into `snapshot-<generation:05d>/`
  - test script supports `--snapshot <generation>`
  - topology plots use BT layout and ranked input/output nodes
- Updated `examples/README.md` to include `navigation`.
- Updated `examples/navigation/README.md` with an explicit note that the example is still experimental.
- Committed and pushed:
  - `7a2de3d` (`Add navigation example and mark as experimental`) to `master`.

## 2026-04-28

- Implemented ANJI-style stranded-node pruning in `algorithm_mode = anji`:
  - Added forward/reverse reachability prune pass in `neat/anji_compat.py`.
  - Hidden nodes are structurally removed when unreachable from inputs or unable to reach outputs.
  - Added `anji_prune_rate` (default `1.0`) to control prune-pass probability.
- Added ANJI prune tests in `tests/test_anji_mode.py`:
  - no-input hidden node pruning
  - no-output hidden node pruning
  - self-loop-only hidden node pruning
- Updated LunarLander ANJI snapshot config:
  - `examples/lunar-lander/config-feedforward-anji4-snapshot`
  - explicitly sets `anji_prune=true` and `anji_prune_rate=1.0`
- Added/updated README files for all example task directories under `examples/` with:
  - evolve/test commands
  - CLI arguments
  - task-specific config knobs
  - expected outputs/runtime directories
- Committed and pushed:
  - `6cf1874` (`Add ANJI stranded-node pruning and refresh example READMEs`) to `master`.
