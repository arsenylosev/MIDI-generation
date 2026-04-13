"""Sparse candidate generator for beat-level transitions.

Implements the gating rules described in the revised strategy note
(Section 2 & Step 3): meter gating, harmonic rhythm gating, region
gating, chord gating, melody gating, and groove gating.

The generator produces a small set of musically legal next states for
any given current state, which the learned scorer then ranks.
"""

from midi_gen.candidates.generator import SparseCandidateGenerator
from midi_gen.candidates.gating import (
    meter_gate,
    harmonic_rhythm_gate,
    region_gate,
    chord_gate,
    melody_gate,
    groove_gate,
)

__all__ = [
    "SparseCandidateGenerator",
    "meter_gate",
    "harmonic_rhythm_gate",
    "region_gate",
    "chord_gate",
    "melody_gate",
    "groove_gate",
]
