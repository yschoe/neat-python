# Export Example

Trains a small XOR network and exports it to a framework-agnostic JSON format.

## Run

```bash
cd examples/export
python export_example.py
```

## Test/inspect

- Inspect exported JSON (default: `xor_winner.json`)
- Use helper tools:

```bash
python neat_analyzer.py xor_winner.json
python neat_to_frameworks.py xor_winner.json
```

## Task-specific config knobs

This example uses an internal XOR setup; key NEAT knobs are the same as the XOR examples:
- structural mutation rates
- compatibility coefficients
- population size and termination threshold

## Outputs

- `xor_winner.json`
- console summary from analyzer/export helpers
