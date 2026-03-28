# ANJI Mode Parity Checklist

This checklist tracks parity between ANJI (`scratch/anji_2_01`) and the newly added
`algorithm_mode = anji` path in this repository.

Status labels:

- `Exact`: Intended to match ANJI behavior closely.
- `Approximate`: Semantically similar but not byte-for-byte or algorithm-for-algorithm identical.
- `Not Implemented`: Known gap.

## A. Mode Selection and Wiring

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Mode selection | Java-side class wiring | `algorithm_mode = anji` in `[NEAT]` auto-swaps default classes | Exact | Requires using default class arguments in `neat.Config(...)` |
| Config section names | `DefaultGenome`/`DefaultReproduction` style INI keys in this repo | ANJI classes use section aliases to preserve existing section names | Exact | No config section rename required |

## B. Species / Stagnation / Threshold

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Species stagnation elimination | No dedicated stagnation-removal module | `AnjiNoStagnation` never marks species stagnant | Exact | Species can still disappear if they become empty |
| Compatibility threshold | Static (unless user changes config externally) | Dynamic targeting disabled in `AnjiSpeciesSet` | Exact | Uses configured `compatibility_threshold` |
| Speciation assignment policy | Representative-based threshold matching | Inherits existing species assignment logic | Approximate | Representative update/order may differ from ANJI internals |

## C. Reproduction Pipeline

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Pipeline shape | Evaluate -> select survivors -> reproduce -> mutate offspring | Implemented in `AnjiReproduction` | Exact | Survivors are carried over; offspring are mutated |
| Survivor budget | `survival.rate` controls survivors | `anji_survival_rate` controls survivor count | Exact | Rounded to nearest integer |
| Elitism | Optional elitism with minimum species size | `anji_elitism` + `anji_elitism_min_species_size` | Exact | Elites are preserved in survivor set |
| Clone/crossover slices | Clone slice = survival rate; crossover slice = 1 - 2*survival | `anji_clone_slice` / `anji_crossover_slice` (`auto` default mirrors ANJI) | Approximate | Rounding and scaling to exact offspring count differs in details |
| Interspecies crossover | Intra-species mating | Intra-species only in ANJI mode | Exact | No interspecies mating path |
| Species offspring allocation | Proportional to species fitness | Proportional to species mean fitness | Approximate | ANJI/JGAP rounding/tie-breaking differs slightly |

## D. Mutation Semantics

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Weight mutation | Mutate connection weights in offspring | Mutates connection genes in offspring | Approximate | Uses neat-python gene attribute mutation model |
| Add connection mutation | Opportunity-based over candidate unconnected pairs | Opportunity-based over candidate keys | Approximate | Candidate filtering/cycle checks based on existing neat-python graph utils |
| Add node mutation | Split connection; reuse node innovation mapping | Split connection with persistent split map in tracker | Approximate | Depends on local genome representation and node id allocation |
| Classic single topological mutation | Optional single topology mutation per offspring | `anji_topology_mutation_classic` supported | Approximate | Probability composition mirrors ANJI intent |
| Remove connection mutation | Multiple strategies (`SKEWED`, `ALL`, `SMALL`) | Uniform opportunity-based removal via `anji_remove_connection_rate` | Not Implemented | Strategy-specific ANJI removers are not ported |
| Prune mutation | Explicit prune operator | `anji_prune` uses dangling-node pruning pass | Approximate | Not a separate forward/backward traversal operator |
| Node mutation | ANJI neuron alleles largely fixed | Default off (`anji_mutate_nodes = false`) | Exact | Can be enabled for experimentation |
| Node deletion operator | No direct node-delete operator | Not used in ANJI mode path | Exact | Consistent with ANJI operator set |

## E. Innovation Number Handling

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Connection innovation reuse | Persistent `(src,dst)->innovation` mapping | Persistent mapping (no generation reset) | Exact | Implemented by `AnjiInnovationTracker` behavior |
| Split-connection node reuse | Persistent `connection->node` mapping | Implemented via split map in tracker | Exact | Reuses node id for repeated split of same innovation id |
| Generation reset behavior | No generation-local dedup reset requirement | `reset_generation()` no-op in ANJI tracker | Exact | Differs intentionally from default neat-python mode |

## F. Distance and Compatibility (Item 8 context)

| Area | ANJI behavior | neat-python ANJI mode | Status | Notes |
|---|---|---|---|---|
| Distance model | Primarily connection-gene + excess/disjoint/common terms; neuron distance effectively zero | Existing neat-python distance model retained | Approximate | This is a known superset behavior and intentionally left as-is |

## G. Config Keys Added for ANJI Mode

These keys are read in ANJI mode (all optional):

- `[NEAT]`
  - `algorithm_mode = anji`
- `[DefaultReproduction]`
  - `anji_survival_rate`
  - `anji_elitism`
  - `anji_elitism_min_species_size`
  - `anji_clone_slice` (`auto` or float)
  - `anji_crossover_slice` (`auto` or float)
- `[DefaultGenome]`
  - `anji_topology_mutation_classic`
  - `anji_mutate_nodes`
  - `anji_prune`
  - `anji_remove_connection_rate`

## H. Outstanding Gaps (Highest Priority)

1. Port ANJI remove-connection strategies (`SKEWED`, `ALL`, `SMALL`) as first-class options.
2. Implement ANJI-style prune operator traversal (forward/backward visited-graph prune), separate from current dangling-node prune.
3. Tighten reproduction rounding/tie-break behavior for closer JGAP parity.
4. Add golden parity tests against small deterministic ANJI-like scenarios (species allocation and mutation counts).

