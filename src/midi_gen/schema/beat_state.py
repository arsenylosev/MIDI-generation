"""Beat-level structural state representation.

Each beat in the generated music is described by a fixed set of structural
tokens.  This is the *native* internal language of the planner and scorer,
deliberately richer than a raw piano-roll tensor.

Schema fields (frozen after v0.2.0 — see Section 10, Step 1 of the revised
strategy note):

    meter_token      – encoded time-signature class (e.g. 0=4/4, 1=7/8, 2=5/4, 3=6/8, 4=7/4)
    beat_position    – 0-indexed position within the current measure
    boundary_level   – hierarchical boundary strength (0=none, 1=sub-phrase,
                       2=phrase, 3=section)
    region_label     – section / key-region identifier (index into the form
                       string produced by the GTTM prior)
    chord_label      – integer encoding of the current chord (root * 12 + quality)
    harmonic_role    – functional role in the current key (0=tonic, 1=subdominant,
                       2=dominant, 3=applied-dominant, 4=modal-interchange, ...)
    melodic_head     – MIDI pitch of the most salient melodic note at this beat
                       (0–127, or -1 for rest / continuation)
    groove_token     – categorical groove-family index learned from corpus
                       statistics (e.g. straight-8th, shuffle, half-time, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary sizes (frozen schema)
# ---------------------------------------------------------------------------

METER_VOCAB = 5          # 4/4, 7/8, 5/4, 6/8, 7/4
MAX_BEAT_POSITION = 28   # enough for 7/4 at 16th-note resolution
BOUNDARY_LEVELS = 4      # none, sub-phrase, phrase, section
MAX_REGIONS = 32         # max distinct sections in a form string
CHORD_VOCAB = 108        # 12 roots * 9 quality classes
HARMONIC_ROLES = 8       # tonic, subdominant, dominant, applied-dom, modal, chromatic, pedal, other
GROOVE_VOCAB = 16        # categorical groove families

VOCAB_SIZES: Dict[str, int] = {
    "meter_token": METER_VOCAB,
    "beat_position": MAX_BEAT_POSITION,
    "boundary_level": BOUNDARY_LEVELS,
    "region_label": MAX_REGIONS,
    "chord_label": CHORD_VOCAB,
    "harmonic_role": HARMONIC_ROLES,
    "melodic_head": 129,  # 0-127 pitches + rest (-1 mapped to 128)
    "groove_token": GROOVE_VOCAB,
}


# ---------------------------------------------------------------------------
# Meter helpers
# ---------------------------------------------------------------------------

METER_MAP = {
    (4, 4): 0,
    (7, 8): 1,
    (5, 4): 2,
    (6, 8): 3,
    (7, 4): 4,
}

METER_BEATS = {
    0: 4,   # 4/4 → 4 beats
    1: 7,   # 7/8 → 7 eighth-note beats
    2: 5,   # 5/4 → 5 beats
    3: 6,   # 6/8 → 6 eighth-note beats
    4: 7,   # 7/4 → 7 beats
}

INV_METER_MAP = {v: k for k, v in METER_MAP.items()}


# ---------------------------------------------------------------------------
# Chord helpers
# ---------------------------------------------------------------------------

QUALITY_NAMES = ["maj", "min", "dom7", "maj7", "min7", "dim", "aug", "sus4", "sus2"]
ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def encode_chord(root: int, quality: int) -> int:
    """Encode a chord as root * 9 + quality."""
    return root * len(QUALITY_NAMES) + quality


def decode_chord(chord_label: int) -> tuple[int, int]:
    """Decode a chord label into (root, quality)."""
    quality = chord_label % len(QUALITY_NAMES)
    root = chord_label // len(QUALITY_NAMES)
    return root, quality


def chord_name(chord_label: int) -> str:
    """Human-readable chord name."""
    root, quality = decode_chord(chord_label)
    return f"{ROOT_NAMES[root]}{QUALITY_NAMES[quality]}"


# ---------------------------------------------------------------------------
# BeatLevelState
# ---------------------------------------------------------------------------

@dataclass
class BeatLevelState:
    """A single beat's structural description.

    All fields are plain integers so the state can be trivially serialised,
    hashed, and used as a dictionary key or tensor row.
    """

    meter_token: int = 0
    beat_position: int = 0
    boundary_level: int = 0
    region_label: int = 0
    chord_label: int = 0       # encode_chord(root, quality)
    harmonic_role: int = 0
    melodic_head: int = -1     # -1 = rest / continuation
    groove_token: int = 0

    # --- convenience ---

    def to_vector(self) -> np.ndarray:
        """Return a 1-D int array of length 8."""
        mh = self.melodic_head if self.melodic_head >= 0 else 128
        return np.array([
            self.meter_token,
            self.beat_position,
            self.boundary_level,
            self.region_label,
            self.chord_label,
            self.harmonic_role,
            mh,
            self.groove_token,
        ], dtype=np.int32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "BeatLevelState":
        mh = int(vec[6])
        if mh == 128:
            mh = -1
        return cls(
            meter_token=int(vec[0]),
            beat_position=int(vec[1]),
            boundary_level=int(vec[2]),
            region_label=int(vec[3]),
            chord_label=int(vec[4]),
            harmonic_role=int(vec[5]),
            melodic_head=mh,
            groove_token=int(vec[7]),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure all values are native Python types for JSON serialization
        return {k: int(v) if isinstance(v, (np.integer,)) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BeatLevelState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# BeatLevelSequence
# ---------------------------------------------------------------------------

@dataclass
class BeatLevelSequence:
    """An ordered sequence of beat-level states forming a structural trajectory.

    This is the primary output of the planner (GTTM prior + SB solver +
    candidate scorer) and the primary input to the multitrack realizer.
    """

    states: List[BeatLevelState] = field(default_factory=list)

    # Metadata carried alongside the trajectory
    bpm: float = 120.0
    key: int = 0          # pitch-class of tonic (0=C)
    is_major: bool = True
    genre: str = "prog_rock"
    form_string: str = ""

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    def __iter__(self):
        return iter(self.states)

    def to_matrix(self) -> np.ndarray:
        """Return (N, 8) int32 matrix."""
        if not self.states:
            return np.zeros((0, 8), dtype=np.int32)
        return np.stack([s.to_vector() for s in self.states])

    @classmethod
    def from_matrix(cls, mat: np.ndarray, **kwargs) -> "BeatLevelSequence":
        states = [BeatLevelState.from_vector(mat[i]) for i in range(mat.shape[0])]
        return cls(states=states, **kwargs)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise to JSON."""
        data = {
            "bpm": self.bpm,
            "key": self.key,
            "is_major": self.is_major,
            "genre": self.genre,
            "form_string": self.form_string,
            "states": [s.to_dict() for s in self.states],
        }
        text = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_str: str) -> "BeatLevelSequence":
        """Deserialise from JSON file or string."""
        try:
            with open(path_or_str) as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            data = json.loads(path_or_str)
        states = [BeatLevelState.from_dict(d) for d in data.get("states", [])]
        return cls(
            states=states,
            bpm=data.get("bpm", 120.0),
            key=data.get("key", 0),
            is_major=data.get("is_major", True),
            genre=data.get("genre", "prog_rock"),
            form_string=data.get("form_string", ""),
        )
