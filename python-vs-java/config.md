# ANJI/Fork Configuration Reference

This document summarizes fork-specific configuration keys related to ANJI compatibility and recent workflow additions.

## 1. Core Mode Selection

- Section: `[NEAT]`
- Key: `algorithm_mode`
- Values:
  - `default` (or omitted): standard neat-python behavior.
  - `anji`: enables ANJI-compatibility class wiring for genome/species/reproduction/stagnation internals.

## 2. Snapshot Support (Fork Addition)

- Section: `[NEAT]`
- Key: `snapshot_interval`
- Type: integer
- Default: `100` if omitted.
- Meaning:
  - During evolution, every `snapshot_interval` generations, the current best genome/artifacts are saved under:
    - `exp-<config-filename>/snapshot-<generation:05d>/`
  - Example: generation 100 -> `snapshot-00100`.
- Notes:
  - Final generation artifacts are still written directly in `exp-<config-filename>/` as before.
  - Snapshot plot rendering is non-blocking (`view=False`).

## 3. ANJI Genome Compatibility Keys

- Section: `[DefaultGenome]` (or `[IZGenome]` where relevant)

### `anji_topology_mutation_classic`
- Type: boolean
- Default: `false`
- Meaning:
  - `true`: use ANJI-style single topological mutation opportunity (add-node vs add-connection choice logic).
  - `false`: use opportunity-based topology mutation over candidates.

### `anji_mutate_nodes`
- Type: boolean
- Default: `false`
- Meaning:
  - Controls whether node genes are mutated in ANJI mode.
  - Typically kept `false` to stay closer to ANJI behavior.

### `anji_prune`
- Type: boolean
- Default: `true`
- Meaning:
  - Enables pruning of hidden nodes that cannot affect outputs after mutation.

### `anji_remove_connection_rate`
- Type: float
- Default behavior:
  - Uses configured value if present.
  - Otherwise falls back to `conn_delete_prob`.
- Meaning:
  - Probability used for ANJI-style connection removal opportunity in offspring mutation.

## 4. ANJI Reproduction Compatibility Keys

- Section: `[DefaultReproduction]`

### `anji_survival_rate`
- Type: float
- Default: `0.2`
- Meaning:
  - Fraction of population retained as survivors before offspring creation.

### `anji_elitism`
- Type: boolean
- Default: `true`
- Meaning:
  - Preserves per-species elites (subject to species-size threshold).

### `anji_elitism_min_species_size`
- Type: integer
- Default: `6`
- Meaning:
  - Minimum species size required for ANJI-style elite preservation.

### `anji_clone_slice`
- Type: `auto` or float string
- Default: `auto`
- Meaning:
  - Fraction of offspring budget produced via cloning.
  - `auto` aligns with ANJI-inspired slice behavior.

### `anji_crossover_slice`
- Type: `auto` or float string
- Default: `auto`
- Meaning:
  - Fraction of offspring budget produced via crossover.
  - `auto` aligns with ANJI-inspired slice behavior.

## 5. Test Script Snapshot Selection

For `examples/*/test-feedforward.py` scripts:

- CLI option: `--snapshot <generation>`
- Meaning:
  - Load winner genome from:
    - `exp-<config-filename>/snapshot-<generation:05d>/...`
  - If not provided, scripts keep existing behavior (final winner in run directory / cwd fallback where implemented).

## 6. Practical Notes

- If you are tuning ANJI mode, keep `algorithm_mode = anji` and explicitly set the `anji_*` keys in config files for clarity.
- If runs appear slow at higher generations, consider lowering topology mutation rates and/or adjusting pruning-related knobs.
- `snapshot_interval` can be increased for long runs to reduce I/O overhead.
