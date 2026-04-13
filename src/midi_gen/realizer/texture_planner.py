"""Texture planner: assigns bar-level arrangement codes from structural context.

Given a BeatLevelSequence, the texture planner groups beats into bars and
assigns a BarTextureCode to each bar.  The assignment is driven by:

    - The region label (verse vs. chorus vs. bridge → different textures)
    - The tension / energy curve from the GTTM prior
    - The groove token (determines drum pattern family)
    - Genre-specific arrangement conventions

This is a rule-based planner that can later be replaced or augmented by
a learned model (Phase 2 of the build order).
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence, METER_BEATS
from midi_gen.schema.bar_texture import (
    BarTextureCode,
    ArrangementPlan,
    TrackRole,
)


# ---------------------------------------------------------------------------
# Genre-specific texture mappings
# ---------------------------------------------------------------------------

# Maps (region_type, energy_level) → default texture codes
# energy_level: "low" (<0.3), "mid" (0.3–0.7), "high" (>0.7)

PROG_ROCK_TEXTURES: Dict[str, Dict[str, Dict[str, int]]] = {
    "intro": {
        "low":  {"drums": 15, "bass": 14, "comping": 3, "lead": 13, "aux": 0},
        "mid":  {"drums": 0,  "bass": 0,  "comping": 3, "lead": 0,  "aux": 0},
        "high": {"drums": 1,  "bass": 4,  "comping": 0, "lead": 0,  "aux": 0},
    },
    "verse": {
        "low":  {"drums": 0,  "bass": 0,  "comping": 4, "lead": 0,  "aux": 0},
        "mid":  {"drums": 0,  "bass": 2,  "comping": 0, "lead": 3,  "aux": 0},
        "high": {"drums": 1,  "bass": 4,  "comping": 2, "lead": 5,  "aux": 0},
    },
    "chorus": {
        "low":  {"drums": 0,  "bass": 1,  "comping": 0, "lead": 0,  "aux": 0},
        "mid":  {"drums": 4,  "bass": 4,  "comping": 5, "lead": 3,  "aux": 0},
        "high": {"drums": 4,  "bass": 4,  "comping": 5, "lead": 5,  "aux": 1},
    },
    "bridge": {
        "low":  {"drums": 3,  "bass": 3,  "comping": 7, "lead": 12, "aux": 0},
        "mid":  {"drums": 10, "bass": 6,  "comping": 2, "lead": 6,  "aux": 0},
        "high": {"drums": 10, "bass": 9,  "comping": 2, "lead": 5,  "aux": 1},
    },
    "solo": {
        "low":  {"drums": 8,  "bass": 2,  "comping": 7, "lead": 3,  "aux": 0},
        "mid":  {"drums": 8,  "bass": 2,  "comping": 1, "lead": 6,  "aux": 0},
        "high": {"drums": 4,  "bass": 9,  "comping": 2, "lead": 5,  "aux": 1},
    },
    "outro": {
        "low":  {"drums": 3,  "bass": 3,  "comping": 3, "lead": 0,  "aux": 0},
        "mid":  {"drums": 0,  "bass": 0,  "comping": 3, "lead": 13, "aux": 0},
        "high": {"drums": 0,  "bass": 4,  "comping": 0, "lead": 0,  "aux": 0},
    },
}

JAZZ_FUSION_TEXTURES: Dict[str, Dict[str, Dict[str, int]]] = {
    "intro": {
        "low":  {"drums": 9,  "bass": 14, "comping": 7, "lead": 12, "aux": 0},
        "mid":  {"drums": 8,  "bass": 2,  "comping": 7, "lead": 0,  "aux": 0},
        "high": {"drums": 8,  "bass": 2,  "comping": 1, "lead": 3,  "aux": 0},
    },
    "head": {
        "low":  {"drums": 8,  "bass": 2,  "comping": 7, "lead": 0,  "aux": 0},
        "mid":  {"drums": 8,  "bass": 2,  "comping": 1, "lead": 3,  "aux": 0},
        "high": {"drums": 7,  "bass": 9,  "comping": 5, "lead": 5,  "aux": 0},
    },
    "solo": {
        "low":  {"drums": 9,  "bass": 2,  "comping": 7, "lead": 3,  "aux": 0},
        "mid":  {"drums": 8,  "bass": 2,  "comping": 1, "lead": 6,  "aux": 0},
        "high": {"drums": 7,  "bass": 9,  "comping": 2, "lead": 5,  "aux": 1},
    },
    "interlude": {
        "low":  {"drums": 9,  "bass": 14, "comping": 3, "lead": 12, "aux": 0},
        "mid":  {"drums": 18, "bass": 6,  "comping": 7, "lead": 0,  "aux": 0},
        "high": {"drums": 7,  "bass": 4,  "comping": 2, "lead": 6,  "aux": 0},
    },
    "outro": {
        "low":  {"drums": 9,  "bass": 14, "comping": 3, "lead": 13, "aux": 0},
        "mid":  {"drums": 8,  "bass": 2,  "comping": 7, "lead": 0,  "aux": 0},
        "high": {"drums": 8,  "bass": 2,  "comping": 1, "lead": 3,  "aux": 0},
    },
}

GENRE_TEXTURE_MAPS = {
    "prog_rock": PROG_ROCK_TEXTURES,
    "jazz_fusion": JAZZ_FUSION_TEXTURES,
}

# Section name aliases for form string parsing
SECTION_ALIASES = {
    "A": "verse", "B": "chorus", "C": "bridge", "D": "solo",
    "I": "intro", "O": "outro", "S": "solo", "H": "head",
    "V": "verse", "Ch": "chorus", "Br": "bridge", "Int": "interlude",
}


class TexturePlanner:
    """Assign bar-level texture codes from a beat-level structural trajectory.

    Parameters
    ----------
    genre : str
        Target genre (selects the texture mapping table).
    """

    def __init__(self, genre: str = "prog_rock"):
        self.genre = genre
        self.texture_map = GENRE_TEXTURE_MAPS.get(genre, PROG_ROCK_TEXTURES)

    def plan(
        self,
        sequence: BeatLevelSequence,
        tension_curve: Optional[np.ndarray] = None,
    ) -> ArrangementPlan:
        """Generate an ArrangementPlan for the given structural trajectory.

        Parameters
        ----------
        sequence : BeatLevelSequence
            The beat-level structural trajectory.
        tension_curve : np.ndarray, optional
            Per-beat tension values (0–1). If None, derived from boundary levels.

        Returns
        -------
        ArrangementPlan
            One BarTextureCode per bar.
        """
        bars = self._group_into_bars(sequence)

        if tension_curve is None:
            tension_curve = self._derive_tension(sequence)

        plan_bars = []
        for bar_idx, (beat_start, beat_end) in enumerate(bars):
            # Representative state for this bar (first beat)
            state = sequence[beat_start]

            # Compute average tension for this bar
            bar_tension = float(np.mean(tension_curve[beat_start:beat_end]))

            # Determine section type from region label and form string
            section = self._resolve_section(state.region_label, sequence.form_string)

            # Look up texture codes
            energy_level = "low" if bar_tension < 0.3 else ("high" if bar_tension > 0.7 else "mid")
            texture = self._lookup_texture(section, energy_level)

            plan_bars.append(BarTextureCode(
                bar_index=bar_idx,
                drums=texture["drums"],
                bass=texture["bass"],
                comping=texture["comping"],
                lead=texture["lead"],
                aux=texture.get("aux", 0),
                energy=bar_tension,
            ))

        return ArrangementPlan(bars=plan_bars, genre=self.genre)

    def _group_into_bars(
        self, sequence: BeatLevelSequence,
    ) -> List[tuple[int, int]]:
        """Group beats into bars based on beat_position resets."""
        bars = []
        bar_start = 0

        for i in range(1, len(sequence)):
            if sequence[i].beat_position == 0:
                bars.append((bar_start, i))
                bar_start = i

        # Last bar
        if bar_start < len(sequence):
            bars.append((bar_start, len(sequence)))

        return bars

    def _derive_tension(self, sequence: BeatLevelSequence) -> np.ndarray:
        """Derive a tension curve from boundary levels and harmonic roles."""
        n = len(sequence)
        tension = np.zeros(n, dtype=np.float32)

        for i, state in enumerate(sequence):
            # Base tension from harmonic role
            role_tension = {0: 0.2, 1: 0.4, 2: 0.7, 3: 0.8, 4: 0.6}.get(
                state.harmonic_role, 0.5
            )
            # Boundary boost
            boundary_boost = state.boundary_level * 0.15

            tension[i] = min(1.0, role_tension + boundary_boost)

        # Smooth with a small window
        if n > 4:
            kernel = np.ones(4) / 4
            tension = np.convolve(tension, kernel, mode="same")

        return np.clip(tension, 0.0, 1.0)

    def _resolve_section(self, region_label: int, form_string: str) -> str:
        """Map a region label to a section type name."""
        if form_string:
            parts = [p.strip() for p in form_string.replace("-", " ").split()]
            if region_label < len(parts):
                token = parts[region_label]
                return SECTION_ALIASES.get(token, token.lower())

        # Default mapping by region index
        default_sections = ["intro", "verse", "chorus", "bridge", "solo", "outro"]
        return default_sections[region_label % len(default_sections)]

    def _lookup_texture(self, section: str, energy_level: str) -> Dict[str, int]:
        """Look up texture codes from the genre table."""
        section_map = self.texture_map.get(section)
        if section_map is None:
            # Fallback to verse
            section_map = self.texture_map.get("verse", self.texture_map.get(
                list(self.texture_map.keys())[0]
            ))
        return section_map.get(energy_level, section_map.get("mid", {}))
