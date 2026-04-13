"""Sparse candidate generator.

Given a current BeatLevelState and the structural context (phrase structure,
tension curve, key), this module generates a small set of musically legal
next states by:

    1. Enumerating plausible next states from rules and corpus lookups
    2. Applying the six gating filters sequentially
    3. Returning the surviving candidates for the learned scorer to rank

The generator is deliberately simple and inspectable (revised strategy note,
Section 10, Step 3).
"""

from __future__ import annotations

import random
from typing import List, Optional, Dict, Any

import numpy as np

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    METER_BEATS,
    METER_MAP,
    CHORD_VOCAB,
    HARMONIC_ROLES,
    GROOVE_VOCAB,
    encode_chord,
    decode_chord,
)
from midi_gen.candidates.gating import (
    meter_gate,
    harmonic_rhythm_gate,
    region_gate,
    chord_gate,
    melody_gate,
    groove_gate,
)


class SparseCandidateGenerator:
    """Generates a small set of musically legal next beat-level states.

    Parameters
    ----------
    genre : str
        Target genre (affects interval constraints, groove families, etc.).
    max_candidates : int
        Maximum number of candidates to return after gating.
    seed : int or None
        Random seed for reproducibility.
    """

    # Genre-specific configuration
    GENRE_CONFIG: Dict[str, Dict[str, Any]] = {
        "prog_rock": {
            "max_melody_interval": 12,
            "allowed_meters": [0, 1, 2, 3],  # 4/4, 7/8, 5/4, 6/8
            "chord_distance": 3,
            "groove_families": list(range(12)),
            "melodic_range": (48, 84),  # C3 to C6
        },
        "jazz_fusion": {
            "max_melody_interval": 14,  # wider intervals in jazz
            "allowed_meters": [0, 2, 4],  # 4/4, 5/4, 7/4
            "chord_distance": 4,  # more distant chords allowed
            "groove_families": list(range(16)),
            "melodic_range": (52, 88),  # E3 to E6
        },
    }

    def __init__(
        self,
        genre: str = "prog_rock",
        max_candidates: int = 16,
        seed: Optional[int] = None,
    ):
        self.genre = genre
        self.max_candidates = max_candidates
        self.config = self.GENRE_CONFIG.get(genre, self.GENRE_CONFIG["prog_rock"])
        self.rng = random.Random(seed)

    def generate(
        self,
        current: BeatLevelState,
        context: Optional[BeatLevelSequence] = None,
        tension_target: float = 0.5,
        phrase_structure: Optional[List[Dict]] = None,
    ) -> List[BeatLevelState]:
        """Generate gated candidates for the next beat.

        Parameters
        ----------
        current : BeatLevelState
            The current beat-level state.
        context : BeatLevelSequence, optional
            The trajectory so far (for context-aware generation).
        tension_target : float
            Target tension level from the GTTM prolongational curve (0–1).
        phrase_structure : list of dict, optional
            Phrase structure from the GTTM prior.

        Returns
        -------
        list of BeatLevelState
            A small set of musically legal next states (at most max_candidates).
        """
        # Step 1: Enumerate raw candidates
        raw = self._enumerate_candidates(current, tension_target)

        # Step 2: Apply gating filters sequentially
        gated = meter_gate(current, raw)
        gated = harmonic_rhythm_gate(current, gated)
        gated = region_gate(current, gated)
        gated = chord_gate(current, gated, max_distance=self.config["chord_distance"])
        gated = melody_gate(
            current, gated, max_interval=self.config["max_melody_interval"]
        )
        gated = groove_gate(current, gated)

        # Step 3: Trim to max_candidates (random sample if too many)
        if len(gated) > self.max_candidates:
            gated = self.rng.sample(gated, self.max_candidates)

        # Ensure at least one candidate (fallback: hold current state)
        if not gated:
            gated = [self._hold_candidate(current)]

        return gated

    def _enumerate_candidates(
        self,
        current: BeatLevelState,
        tension_target: float,
    ) -> List[BeatLevelState]:
        """Enumerate plausible next states before gating.

        This is the 'raw proposal' step.  It generates candidates by
        varying each field independently and combining common patterns.
        """
        candidates = []
        beats = METER_BEATS.get(current.meter_token, 4)
        next_pos = (current.beat_position + 1) % beats
        is_barline = next_pos == 0

        # --- Beat position candidates ---
        positions = [next_pos]
        if is_barline:
            positions = [0]  # always reset to 0 at barline

        # --- Boundary candidates ---
        boundary_levels = [0]  # default: no boundary
        if is_barline:
            boundary_levels.extend([1, 2])  # sub-phrase or phrase boundary possible
            if self.rng.random() < 0.1:
                boundary_levels.append(3)  # rare section boundary

        # --- Chord candidates ---
        current_root, current_qual = decode_chord(current.chord_label)
        chord_candidates = [current.chord_label]  # hold current chord

        if is_barline or next_pos in {0, 2}:
            # On strong beats, propose chord changes
            for interval in [0, 5, 7, 2, 10, 3, 4, 9]:  # common root motions
                new_root = (current_root + interval) % 12
                for qual in self._tension_appropriate_qualities(tension_target):
                    chord_candidates.append(encode_chord(new_root, qual))

        # --- Melodic head candidates ---
        mel_range = self.config["melodic_range"]
        melody_candidates = []
        if current.melodic_head >= 0:
            # Stepwise motion
            for step in [-2, -1, 0, 1, 2]:
                p = current.melodic_head + step
                if mel_range[0] <= p <= mel_range[1]:
                    melody_candidates.append(p)
            # Thirds and fourths
            for leap in [-3, 3, -4, 4, -5, 5, 7, -7, 12, -12]:
                p = current.melodic_head + leap
                if mel_range[0] <= p <= mel_range[1]:
                    melody_candidates.append(p)
            # Rest
            melody_candidates.append(-1)
        else:
            # Coming from rest: propose common starting pitches
            melody_candidates = [-1]  # stay on rest
            for p in range(mel_range[0], mel_range[1], 4):
                melody_candidates.append(p)

        # Deduplicate
        melody_candidates = list(set(melody_candidates))

        # --- Harmonic role candidates ---
        role_candidates = self._tension_appropriate_roles(tension_target)

        # --- Groove candidates ---
        groove_candidates = [current.groove_token]
        if is_barline:
            for g in self.config["groove_families"]:
                if abs(g - current.groove_token) <= 2:
                    groove_candidates.append(g)
        groove_candidates = list(set(groove_candidates))

        # --- Combine into BeatLevelState candidates ---
        # Don't do full cartesian product — sample representative combinations
        for pos in positions:
            for boundary in boundary_levels:
                for chord in chord_candidates[:8]:  # limit chord variety
                    for mel in melody_candidates[:6]:  # limit melody variety
                        role = self.rng.choice(role_candidates)
                        groove = self.rng.choice(groove_candidates)
                        candidates.append(BeatLevelState(
                            meter_token=current.meter_token,
                            beat_position=pos,
                            boundary_level=boundary,
                            region_label=current.region_label,
                            chord_label=chord,
                            harmonic_role=role,
                            melodic_head=mel,
                            groove_token=groove,
                        ))

        # Deduplicate by converting to tuples
        seen = set()
        unique = []
        for c in candidates:
            key = tuple(c.to_vector().tolist())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _hold_candidate(self, current: BeatLevelState) -> BeatLevelState:
        """Create a 'hold' candidate that advances only the beat position."""
        beats = METER_BEATS.get(current.meter_token, 4)
        return BeatLevelState(
            meter_token=current.meter_token,
            beat_position=(current.beat_position + 1) % beats,
            boundary_level=0,
            region_label=current.region_label,
            chord_label=current.chord_label,
            harmonic_role=current.harmonic_role,
            melodic_head=current.melodic_head,
            groove_token=current.groove_token,
        )

    def _tension_appropriate_qualities(self, tension: float) -> List[int]:
        """Return chord quality indices appropriate for the tension level."""
        if tension < 0.3:
            return [0, 1, 3, 4]       # maj, min, maj7, min7 (stable)
        elif tension < 0.6:
            return [0, 1, 2, 3, 4, 7]  # add dom7, sus4
        else:
            return [0, 1, 2, 4, 5, 6, 7, 8]  # add dim, aug, sus2 (tense)

    def _tension_appropriate_roles(self, tension: float) -> List[int]:
        """Return harmonic role indices appropriate for the tension level."""
        if tension < 0.3:
            return [0, 1]          # tonic, subdominant
        elif tension < 0.6:
            return [0, 1, 2]       # add dominant
        else:
            return [0, 1, 2, 3, 4]  # add applied-dominant, modal-interchange
