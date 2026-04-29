# Picture2D / CPPN

Evolves image-generating CPPNs (color or grayscale), including novelty and interactive modes.

## Evolve

Novelty mode:
```bash
cd examples/picture2d
python evolve_novelty.py [novelty_config*]
```

Interactive mode:
```bash
python evolve_interactive.py [interactive_config_*]
```

## Render/Test

```bash
python render.py
python render_genome.py
```

## Task-specific config knobs

- image resolution and channel settings
- novelty parameters (novelty config files)
- activation/aggregation options for CPPN behavior
- `snapshot_interval` where applicable

## Outputs

Runtime dirs: `exp-<config-filename>/`
- evolved genomes/checkpoints
- rendered images
- fitness/novelty/species plots (mode-dependent)
