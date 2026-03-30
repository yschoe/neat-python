# Navigation example

This example evolves a feed-forward controller for a 2D navigation task.
An agent must move from a random start to a random target in a square arena.

> Note: this example is still in an experimental stage and may change.

The controller outputs:

- `thrust`: move forward when positive
- `turn`: rotate heading left/right

Optional barrier mode can place an obstacle line segment between start and target.

## Run evolution

```bash
cd examples/navigation
python evolve-feedforward.py
```

Use ANJI mode config:

```bash
python evolve-feedforward.py config-feedforward-anji
```

Optional barrier:

```bash
python evolve-feedforward.py config-feedforward --barrier-length 100
```

Results are written to:

- `exp-config-feedforward/`
- `exp-config-feedforward-anji/`

Snapshots are saved at `snapshot_interval` generations:

- `exp-<config-filename>/snapshot-00100/`, etc.

## Test winner

```bash
python test-feedforward.py
```

With a specific snapshot:

```bash
python test-feedforward.py config-feedforward --snapshot 100
```
