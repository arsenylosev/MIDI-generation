"""Gating rules for the sparse candidate generator.

Each gate is a filter that removes musically illegal or implausible
transitions from the candidate set.  Gates are applied sequentially:
the output of one gate becomes the input of the next.

The gates are deliberately simple and inspectable — you should be able
to print the candidate set for any given state and understand why each
candidate was allowed.  This is a core design principle from the revised
strategy note (Section 10, Step 3).

Gate reference (from the December note, Section 7.2):
    1. Meter gating       – beat_position must follow the current meter
    2. Harmonic rhythm     – chord changes only on metrically strong beats
    3. Region gating       – region_label changes only at phrase boundaries
    4. Chord gating        – next chord must be tonally reachable
    5. Melody gating       – melodic intervals constrained by style
    6. Groove gating       – groove changes only at section boundaries
"""

from __future__ import annotations

from typing import List

from midi_gen.schema.beat_state import (
    BeatLevelState,
    METER_BEATS,
    CHORD_VOCAB,
    decode_chord,
)


# ---------------------------------------------------------------------------
# 1. Meter gating
# ---------------------------------------------------------------------------

def meter_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
) -> List[BeatLevelState]:
    """Ensure beat_position advances correctly within the current meter.

    The next beat_position should be (current + 1) mod beats_in_meter,
    unless a boundary forces a meter change.
    """
    beats = METER_BEATS.get(current.meter_token, 4)
    expected_pos = (current.beat_position + 1) % beats

    result = []
    for c in candidates:
        # Allow the candidate if it has the expected beat position
        # OR if it's a boundary beat (boundary_level >= 2) where meter can change
        if c.beat_position == expected_pos:
            result.append(c)
        elif c.boundary_level >= 2 and c.beat_position == 0:
            # Section/phrase boundary: allow reset to beat 0 with any meter
            result.append(c)

    # If gating is too strict and removes everything, relax to allow any beat 0
    if not result:
        result = [c for c in candidates if c.beat_position == 0 or c.beat_position == expected_pos]
    if not result:
        result = candidates  # fallback: don't gate at all

    return result


# ---------------------------------------------------------------------------
# 2. Harmonic rhythm gating
# ---------------------------------------------------------------------------

def harmonic_rhythm_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
) -> List[BeatLevelState]:
    """Chord changes should only occur on metrically strong beats.

    Strong beats are defined as beat_position == 0 (downbeat) or at
    half-bar positions.  On weak beats, the chord must stay the same.
    """
    beats = METER_BEATS.get(current.meter_token, 4)
    next_pos = (current.beat_position + 1) % beats

    # Define strong positions for each meter
    strong_positions = _strong_positions(current.meter_token)

    result = []
    for c in candidates:
        if c.beat_position in strong_positions:
            # Strong beat: any chord is allowed
            result.append(c)
        elif c.chord_label == current.chord_label:
            # Weak beat: chord must stay the same
            result.append(c)
        elif c.boundary_level >= 2:
            # Phrase boundary overrides harmonic rhythm constraints
            result.append(c)

    if not result:
        result = candidates
    return result


def _strong_positions(meter_token: int) -> set:
    """Return the set of metrically strong beat positions for a meter."""
    mapping = {
        0: {0, 2},        # 4/4: beats 1 and 3
        1: {0, 3, 5},     # 7/8: grouped as 2+2+3 or 3+2+2
        2: {0, 2, 3},     # 5/4: grouped as 3+2 or 2+3
        3: {0, 3},        # 6/8: beats 1 and 4
        4: {0, 2, 4},     # 7/4: grouped as 4+3 or 3+4
    }
    return mapping.get(meter_token, {0})


# ---------------------------------------------------------------------------
# 3. Region gating
# ---------------------------------------------------------------------------

def region_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
) -> List[BeatLevelState]:
    """Region (section) changes only at phrase or section boundaries.

    If the current state is not at a boundary, the next state must keep
    the same region_label.
    """
    result = []
    for c in candidates:
        if current.boundary_level >= 2:
            # At a phrase/section boundary: any region is allowed
            result.append(c)
        elif c.region_label == current.region_label:
            # Not at boundary: must stay in same region
            result.append(c)

    if not result:
        result = candidates
    return result


# ---------------------------------------------------------------------------
# 4. Chord gating
# ---------------------------------------------------------------------------

# Circle of fifths distance matrix (precomputed)
_FIFTH_DISTANCE = {}
for _r in range(12):
    for _r2 in range(12):
        _d = min((_r2 - _r) % 12, (_r - _r2) % 12)
        _fifth = min((_r2 - _r) % 12 // 7, (_r - _r2) % 12 // 7)  # rough
        _FIFTH_DISTANCE[(_r, _r2)] = min(_d, 7 - abs(_d - 7))


def chord_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
    max_distance: int = 3,
) -> List[BeatLevelState]:
    """Next chord must be tonally reachable from the current chord.

    Uses a simplified circle-of-fifths distance: the root of the next
    chord must be within ``max_distance`` fifths of the current root.
    At boundaries, the constraint is relaxed.
    """
    current_root, _ = decode_chord(current.chord_label)

    result = []
    for c in candidates:
        next_root, _ = decode_chord(c.chord_label)
        dist = _circle_of_fifths_distance(current_root, next_root)

        if dist <= max_distance:
            result.append(c)
        elif c.boundary_level >= 2:
            # Boundaries allow distant modulations
            result.append(c)

    if not result:
        result = candidates
    return result


def _circle_of_fifths_distance(root1: int, root2: int) -> int:
    """Compute the shortest distance on the circle of fifths."""
    # Each step on the circle of fifths is +7 semitones (mod 12)
    forward = 0
    r = root1
    while r != root2 and forward < 12:
        r = (r + 7) % 12
        forward += 1
    backward = 0
    r = root1
    while r != root2 and backward < 12:
        r = (r - 7) % 12
        backward += 1
    return min(forward, backward)


# ---------------------------------------------------------------------------
# 5. Melody gating
# ---------------------------------------------------------------------------

def melody_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
    max_interval: int = 12,
    style_intervals: dict | None = None,
) -> List[BeatLevelState]:
    """Melodic intervals constrained by style.

    By default, allows up to an octave leap.  The ``style_intervals``
    dict can override this per-genre (e.g., jazz fusion allows wider
    intervals than pop).
    """
    if current.melodic_head < 0:
        # Current is a rest — any melodic head is allowed
        return candidates

    result = []
    for c in candidates:
        if c.melodic_head < 0:
            # Rest is always allowed
            result.append(c)
        else:
            interval = abs(c.melodic_head - current.melodic_head)
            if interval <= max_interval:
                result.append(c)

    if not result:
        result = candidates
    return result


# ---------------------------------------------------------------------------
# 6. Groove gating
# ---------------------------------------------------------------------------

class MeterGate:
    """Class wrapper for meter_gate."""
    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return meter_gate(current, candidates)


class HarmonicRhythmGate:
    """Class wrapper for harmonic_rhythm_gate."""
    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return harmonic_rhythm_gate(current, candidates)


class RegionGate:
    """Class wrapper for region_gate."""
    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return region_gate(current, candidates)


class ChordGate:
    """Class wrapper for chord_gate."""
    def __init__(self, max_distance: int = 3):
        self.max_distance = max_distance

    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return chord_gate(current, candidates, max_distance=self.max_distance)


class MelodyGate:
    """Class wrapper for melody_gate."""
    def __init__(self, max_interval: int = 12):
        self.max_interval = max_interval

    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return melody_gate(current, candidates, max_interval=self.max_interval)


class GrooveGate:
    """Class wrapper for groove_gate."""
    def filter(self, current: BeatLevelState, candidates: List[BeatLevelState]) -> List[BeatLevelState]:
        return groove_gate(current, candidates)


def groove_gate(
    current: BeatLevelState,
    candidates: List[BeatLevelState],
) -> List[BeatLevelState]:
    """Groove family changes only at section boundaries (boundary_level >= 3).

    Within a section, the groove token should remain stable.  At phrase
    boundaries (level 2), minor variations are allowed (±1).
    """
    result = []
    for c in candidates:
        if current.boundary_level >= 3:
            # Section boundary: any groove is allowed
            result.append(c)
        elif current.boundary_level >= 2:
            # Phrase boundary: allow nearby grooves
            if abs(c.groove_token - current.groove_token) <= 2:
                result.append(c)
        elif c.groove_token == current.groove_token:
            # Within a phrase: groove must stay the same
            result.append(c)

    if not result:
        result = candidates
    return result
