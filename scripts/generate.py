#!/usr/bin/env python3
"""
Standalone generation script.

Prefer using the installed CLI entry point instead::

    uv run midi-gen --genre prog_rock --num-samples 2

This script is provided for convenience when running outside the
installed package, e.g.::

    uv run python scripts/generate.py --genre prog_rock
"""

from midi_gen.cli import main

if __name__ == "__main__":
    main()
