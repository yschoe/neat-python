#!/usr/bin/env python3
"""
Convert an old-style NEAT Graphviz DOT/GV file to the ANJI-like readable layout.

What this script does:
1. Adds `rankdir=BT` (bottom-to-top) if missing.
2. Detects input nodes and output nodes from node attributes:
   - inputs: `shape=box`
   - outputs: `fillcolor=lightblue`
3. Adds rank groups:
   - `{ rank=min; ... }` for inputs
   - `{ rank=max; ... }` for outputs
4. Writes a new DOT/GV file and renders an SVG with `dot`.

Usage:
    python scratch/tools/dot_relayout.py <input.gv>
    python scratch/tools/dot_relayout.py <input.gv> -o <output.gv>
    python scratch/tools/dot_relayout.py <input.gv> -o <output.gv> --svg <output.svg>
"""

import argparse
import re
import subprocess
from pathlib import Path


NODE_RE = re.compile(r'^\s*([A-Za-z0-9_".-]+)\s*\[(.*?)\]\s*;?\s*$')
ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(".*?"|[^,\s\]]+)')


def parse_attrs(attr_text):
    attrs = {}
    for key, value in ATTR_RE.findall(attr_text):
        attrs[key.lower()] = value.strip().strip('"').lower()
    return attrs


def detect_io_nodes(lines):
    inputs = []
    outputs = []

    for line in lines:
        if '->' in line:
            continue
        if re.match(r'^\s*(graph|node|edge)\s+\[', line):
            continue

        m = NODE_RE.match(line)
        if not m:
            continue

        node_id = m.group(1)
        attrs = parse_attrs(m.group(2))

        if attrs.get('shape') == 'box':
            inputs.append(node_id)
        if attrs.get('fillcolor') == 'lightblue':
            outputs.append(node_id)

    return inputs, outputs


def relayout_dot(text):
    lines = text.splitlines()
    inputs, outputs = detect_io_nodes(lines)

    has_rankdir = bool(re.search(r'\brankdir\s*=', text))
    has_rank_blocks = bool(re.search(r'\brank\s*=\s*(min|max)\b', text))

    out_lines = []
    inserted_rankdir = False

    for line in lines:
        out_lines.append(line)
        if not has_rankdir and not inserted_rankdir and re.search(r'\bdigraph\b.*\{', line):
            out_lines.append('    rankdir=BT;')
            inserted_rankdir = True

    if not has_rank_blocks:
        # Insert rank blocks right before the final top-level closing brace.
        insert_idx = None
        for i in range(len(out_lines) - 1, -1, -1):
            if out_lines[i].strip() == '}':
                insert_idx = i
                break

        if insert_idx is None:
            raise ValueError("Invalid DOT file: missing closing '}'")

        rank_lines = []
        if inputs:
            rank_lines.append('    { rank=min; ' + '; '.join(inputs) + '; }')
        if outputs:
            rank_lines.append('    { rank=max; ' + '; '.join(outputs) + '; }')

        out_lines[insert_idx:insert_idx] = rank_lines

    return '\n'.join(out_lines) + '\n'


def render_svg(dot_path, svg_path):
    subprocess.run(
        ['dot', '-Tsvg', str(dot_path), '-o', str(svg_path)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert old GV/DOT layout to ANJI-like rankdir/rank groups and render SVG.'
    )
    parser.add_argument('input_dot', help='Input .gv/.dot file')
    parser.add_argument(
        '-o', '--output-dot',
        default=None,
        help='Output .gv/.dot path (default: <input>-bt.gv)',
    )
    parser.add_argument(
        '--svg',
        default=None,
        help='Output .svg path (default: same basename as output dot)',
    )
    args = parser.parse_args()

    input_dot = Path(args.input_dot)
    if not input_dot.exists():
        raise FileNotFoundError(f'Input file not found: {input_dot}')

    output_dot = Path(args.output_dot) if args.output_dot else input_dot.with_name(f'{input_dot.stem}-bt{input_dot.suffix}')
    output_svg = Path(args.svg) if args.svg else output_dot.with_suffix('.svg')

    old_text = input_dot.read_text(encoding='utf-8')
    new_text = relayout_dot(old_text)
    output_dot.write_text(new_text, encoding='utf-8')
    render_svg(output_dot, output_svg)

    print(f'Wrote DOT: {output_dot}')
    print(f'Wrote SVG: {output_svg}')


if __name__ == '__main__':
    main()
