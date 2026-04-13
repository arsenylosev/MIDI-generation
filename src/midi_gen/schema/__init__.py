"""Native internal state schema for the GTTM + SB pipeline.

This module defines the beat-level structural tokens and bar-level texture
codes that serve as the project's native modeling language, as recommended
in the revised strategy note (April 2026).

The piano-roll (2, T, 128) format is retained only as an interchange /
compatibility layer for whole-song-gen.  All internal planning, scoring,
and realization operates on these richer representations.
"""

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence
from midi_gen.schema.bar_texture import BarTextureCode, TrackRole
from midi_gen.schema.converters import (
    beat_sequence_to_piano_roll,
    piano_roll_to_beat_sequence,
    beat_sequence_to_midi,
)

__all__ = [
    "BeatLevelState",
    "BeatLevelSequence",
    "BarTextureCode",
    "TrackRole",
    "beat_sequence_to_piano_roll",
    "piano_roll_to_beat_sequence",
    "beat_sequence_to_midi",
]
