# NEAT Algorithm Differences: `neat-python` vs ANJI (Java)

This report compares core NEAT algorithm behavior in:

- Python: `neat/` (this repository)
- Java: `scratch/anji_2_01/src/com/anji/neat` and `scratch/anji_2_01/src/org/jgap`

## 1. Species Lifecycle and Stagnation

- `neat-python` includes explicit species stagnation handling (`max_stagnation`, `species_elitism`) and removes species that do not improve.
- ANJI/JGAP does not implement a separate stagnation module in the core loop; species are culled when they become empty after survivor selection.

## 2. Speciation Threshold Policy

- `neat-python` supports dynamic compatibility-threshold adjustment (`target_num_species`, `threshold_adjust_rate`, min/max clamps).
- ANJI uses a static speciation threshold (`speciation.threshold`) unless changed externally.

## 3. Reproduction Pipeline

- `neat-python` uses one integrated reproduction flow:
  - species fitness adjustment
  - species spawn allocation
  - elitism/survival-threshold parent pool
  - crossover + mutation per child
- ANJI/JGAP uses staged evolution:
  - survivor selection first (`survival.rate`, optional elitism)
  - reproduction operators with fixed slices (clone slice + crossover slice)
  - mutation operators applied to offspring

## 4. Interspecies Crossover

- `neat-python` has configurable interspecies crossover probability (`interspecies_crossover_prob`).
- ANJI crossover is intra-species in `CrossoverReproductionOperator`.

## 5. Fitness Sharing and Spawn Allocation Details

- `neat-python` supports:
  - `fitness_sharing = normalized` (default min/max normalized mean fitness)
  - `fitness_sharing = canonical` (species mean fitness)
  - spawn methods: `smoothed` or direct `proportional`
- ANJI allocates offspring per species by species average fitness and uses survivor selection separately.

## 6. Mutation Semantics

- `neat-python` mutates per genome with configurable node/connection add/delete probabilities; includes node deletion + connection deletion and post-delete dangling-node pruning.
- ANJI mutation operators are opportunity-based in several places (not strictly one mutation trial per genome), with:
  - add-connection
  - add-neuron
  - weight mutation
  - remove-connection (multiple strategies)
  - prune mutation
- ANJI also offers a classic single-topological-mutation mode (`topology.mutation.classic`) via `SingleTopologicalMutationOperator`.

## 7. Innovation Number Handling

- `neat-python` uses `InnovationTracker` with:
  - global counter
  - same-generation deduplication map
  - per-generation reset of deduplication map (`reset_generation`)
- ANJI uses persistent `NeatIdMap` mappings (`connection->neuron`, `(src,dst)->connection`), enabling innovation ID reuse beyond a single generation (and across persisted runs).

## 8. Genome Distance Formula Differences

- `neat-python` distance can include node-gene distance (`compatibility_include_node_genes`) and connection enable/disable mismatch penalty (`compatibility_enable_penalty`).
- ANJI/JGAP distance is allele-set based with excess/disjoint/common terms; neuron allele distance is effectively zero (`NeuronAllele.distance()` returns `0`).

## 9. Initial Topology Initialization

- `neat-python` supports multiple initialization schemes: `unconnected`, `fs_neat*`, `full*`, `partial*` (with direct/nodirect variants).
- ANJI primarily configures initial topology via `initial.topology.fully.connected` and hidden-neuron count.

## 10. Notes on Similarities

Both implementations retain core NEAT principles:

- innovation-number-based historical markings
- compatibility-distance-based speciation
- structural mutation via add-node (split connection) and add-connection
- crossover that aligns homologous genes by innovation history

## High-Level Summary

Compared to ANJI, this `neat-python` codebase currently exposes more configurable species-dynamics behavior in core evolution (stagnation handling, dynamic thresholding, spawn and sharing modes, and optional interspecies crossover), while ANJI uses a more explicitly operator-driven JGAP pipeline with persistent innovation-ID mapping and strong emphasis on configurable mutation operators.
