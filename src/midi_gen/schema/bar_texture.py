"""Bar-level multitrack texture codes.

These codes describe *what kind of arrangement texture* each track should
play at a given bar, without specifying individual notes.  The multitrack
realizer decodes these codes into concrete note events.

Track roles are fixed to the five-part arrangement model recommended in
the revised strategy note (Section 4b):

    drums   – rhythmic pattern / groove family
    bass    – bass motion shape (root, walking, pedal, riff, …)
    comping – harmonic accompaniment density & voicing style
    lead    – melodic contour type (sustained, scalar, arpeggiated, rest, …)
    aux     – optional texture / pad / effect layer
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import List, Dict, Any, Optional

import numpy as np


class TrackRole(IntEnum):
    """Fixed track roles in the arrangement model."""
    DRUMS = 0
    BASS = 1
    COMPING = 2
    LEAD = 3
    AUX = 4


# Vocabulary sizes for each track's texture code
TEXTURE_VOCAB: Dict[TrackRole, int] = {
    TrackRole.DRUMS: 24,    # groove families (straight, shuffle, half-time, fills, …)
    TrackRole.BASS: 16,     # bass motion types
    TrackRole.COMPING: 16,  # comping density / voicing styles
    TrackRole.LEAD: 16,     # melodic contour categories
    TrackRole.AUX: 12,      # texture / pad types
}

# Human-readable names for drum groove families
DRUM_GROOVE_NAMES = [
    "straight_8th", "straight_16th", "shuffle", "half_time",
    "double_time", "6_8_feel", "latin", "funk",
    "jazz_ride", "jazz_brush", "prog_odd", "blast",
    "fill_short", "fill_long", "break", "tacet",
    "tribal", "electronic", "swing_fast", "swing_slow",
    "afrobeat", "reggae", "march", "free",
]

# Human-readable names for bass motion types
BASS_MOTION_NAMES = [
    "root_only", "root_fifth", "walking", "pedal",
    "riff", "arpeggiated", "chromatic_approach", "scalar_run",
    "octave_jump", "syncopated", "slap", "muted",
    "unison_melody", "counter_melody", "drone", "tacet",
]

# Human-readable names for comping styles
COMPING_STYLE_NAMES = [
    "block_chords", "arpeggiated", "rhythmic_stabs", "sustained_pads",
    "sparse_hits", "dense_voicings", "two_hand_spread", "shell_voicings",
    "tremolo", "broken_chords", "power_chords", "fingerpick",
    "strummed", "muted_rhythm", "counter_rhythm", "tacet",
]

# Human-readable names for lead contour types
LEAD_CONTOUR_NAMES = [
    "sustained_note", "scalar_ascend", "scalar_descend", "arpeggiated",
    "repeated_note", "wide_interval", "chromatic", "blues_bend",
    "trill_ornament", "call_response", "unison_riff", "octave_melody",
    "silence", "fade_in", "fade_out", "free",
]


@dataclass
class BarTextureCode:
    """Texture descriptor for one bar of arrangement.

    Each field is a categorical code indexing into the corresponding
    track vocabulary.  The multitrack realizer conditions on these codes
    plus the beat-level structural trajectory to produce concrete MIDI events.
    """

    bar_index: int = 0
    drums: int = 0
    bass: int = 0
    comping: int = 0
    lead: int = 0
    aux: int = 0

    # Optional: energy / density scalar (0.0–1.0) for the whole bar
    energy: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Return a 1-D array [drums, bass, comping, lead, aux, energy*100]."""
        return np.array([
            self.drums, self.bass, self.comping, self.lead, self.aux,
            int(self.energy * 100),
        ], dtype=np.int32)

    @classmethod
    def from_vector(cls, vec: np.ndarray, bar_index: int = 0) -> "BarTextureCode":
        return cls(
            bar_index=bar_index,
            drums=int(vec[0]),
            bass=int(vec[1]),
            comping=int(vec[2]),
            lead=int(vec[3]),
            aux=int(vec[4]),
            energy=float(vec[5]) / 100.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BarTextureCode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> Dict[str, str]:
        """Human-readable description of each track's texture."""
        return {
            "drums": DRUM_GROOVE_NAMES[self.drums] if self.drums < len(DRUM_GROOVE_NAMES) else f"groove_{self.drums}",
            "bass": BASS_MOTION_NAMES[self.bass] if self.bass < len(BASS_MOTION_NAMES) else f"bass_{self.bass}",
            "comping": COMPING_STYLE_NAMES[self.comping] if self.comping < len(COMPING_STYLE_NAMES) else f"comp_{self.comping}",
            "lead": LEAD_CONTOUR_NAMES[self.lead] if self.lead < len(LEAD_CONTOUR_NAMES) else f"lead_{self.lead}",
            "energy": f"{self.energy:.2f}",
        }


@dataclass
class ArrangementPlan:
    """A full arrangement plan: one BarTextureCode per bar.

    Produced by the multitrack realizer's planning stage, consumed by
    its decoding stage to emit concrete MIDI events per track.
    """

    bars: List[BarTextureCode] = field(default_factory=list)
    genre: str = "prog_rock"

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, idx) -> BarTextureCode:
        return self.bars[idx]

    def to_matrix(self) -> np.ndarray:
        """Return (N_bars, 6) matrix."""
        if not self.bars:
            return np.zeros((0, 6), dtype=np.int32)
        return np.stack([b.to_vector() for b in self.bars])

    def energy_curve(self) -> np.ndarray:
        """Return the energy values as a 1-D float array."""
        return np.array([b.energy for b in self.bars], dtype=np.float32)
