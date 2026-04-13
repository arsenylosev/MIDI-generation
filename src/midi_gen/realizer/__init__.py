"""Multitrack bar-level realizer.

Implements the arrangement engine described in the revised strategy note
(Section 4b, Phase 2): given a beat-level structural trajectory, produce
concrete MIDI events for each track role (drums, bass, comping, lead, aux).

The realizer works in two stages:
    1. **Texture planning** — assign a BarTextureCode to each bar based on
       the structural plan (region, tension, groove token).
    2. **Note decoding** — decode each bar's texture code into concrete
       note events for each track, conditioned on the chord and melodic
       context from the beat-level trajectory.
"""

from midi_gen.realizer.texture_planner import TexturePlanner
from midi_gen.realizer.note_decoder import NoteDecoder
from midi_gen.realizer.realizer import MultitrackRealizer

__all__ = ["TexturePlanner", "NoteDecoder", "MultitrackRealizer"]
