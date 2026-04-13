"""Challenge set generator for targeted evaluation.

Implements the evaluation strategy from the revised strategy note (Section 11):
use small, targeted challenge sets rather than large-scale benchmarks.

Challenge sets test specific structural capabilities:
    - Can the model produce a convincing 4-bar phrase with cadence?
    - Can it maintain a groove pattern across 8 bars?
    - Can it modulate between two keys?
    - Can it produce verse-chorus contrast?
    - Can it handle odd meters (5/4, 7/8)?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence, encode_chord


@dataclass
class ChallengeCase:
    """A single challenge test case."""
    name: str
    description: str
    initial_context: BeatLevelSequence
    expected_properties: dict = field(default_factory=dict)
    genre: str = "prog_rock"


@dataclass
class ChallengeResult:
    """Result of running a challenge case."""
    case_name: str
    passed: bool
    score: float
    details: str = ""


class ChallengeSetGenerator:
    """Generate targeted challenge sets for structural evaluation.

    Each challenge provides an initial context and specifies what
    structural properties the continuation should exhibit.
    """

    def generate_all(self, genre: str = "prog_rock") -> List[ChallengeCase]:
        """Generate the full suite of challenge cases."""
        cases = []
        cases.append(self._four_bar_cadence(genre))
        cases.append(self._eight_bar_groove(genre))
        cases.append(self._verse_chorus_contrast(genre))
        cases.append(self._key_modulation(genre))
        cases.append(self._tension_arc(genre))
        if genre == "prog_rock":
            cases.append(self._odd_meter(genre))
        if genre == "jazz_fusion":
            cases.append(self._ii_V_I_resolution(genre))
        return cases

    def _four_bar_cadence(self, genre: str) -> ChallengeCase:
        """Challenge: produce a 4-bar phrase ending with a clear cadence."""
        # Context: 4 bars of I chord, expect continuation with movement and cadence
        states = []
        for bar in range(4):
            for beat in range(4):
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=2 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(0, 0),  # C major
                    harmonic_role=0,
                    melodic_head=60 + (beat % 3),
                    groove_token=0,
                ))

        return ChallengeCase(
            name="four_bar_cadence",
            description="Continue with a 4-bar phrase ending in a V-I cadence",
            initial_context=BeatLevelSequence(states=states, bpm=120, genre=genre),
            expected_properties={
                "min_length": 16,
                "has_cadence": True,
                "ends_on_tonic": True,
            },
            genre=genre,
        )

    def _eight_bar_groove(self, genre: str) -> ChallengeCase:
        """Challenge: maintain a consistent groove pattern for 8 bars."""
        states = []
        groove = 8 if genre == "jazz_fusion" else 0
        for bar in range(4):
            for beat in range(4):
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=2 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(0, 0),
                    harmonic_role=0,
                    melodic_head=-1,
                    groove_token=groove,
                ))

        return ChallengeCase(
            name="eight_bar_groove",
            description="Continue with 8 bars maintaining the same groove family",
            initial_context=BeatLevelSequence(states=states, bpm=120, genre=genre),
            expected_properties={
                "min_length": 32,
                "groove_consistent": True,
            },
            genre=genre,
        )

    def _verse_chorus_contrast(self, genre: str) -> ChallengeCase:
        """Challenge: produce a verse followed by a contrasting chorus."""
        states = []
        # 8-bar verse context
        for bar in range(8):
            for beat in range(4):
                chord = encode_chord(0, 0) if bar % 2 == 0 else encode_chord(7, 0)
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=3 if bar == 0 and beat == 0 else 0,
                    region_label=0,  # verse
                    chord_label=chord,
                    harmonic_role=0 if bar % 2 == 0 else 1,
                    melodic_head=60 + (bar % 4),
                    groove_token=0,
                ))

        return ChallengeCase(
            name="verse_chorus_contrast",
            description="Continue with a chorus that contrasts with the verse",
            initial_context=BeatLevelSequence(
                states=states, bpm=120, genre=genre,
                form_string="A B",
            ),
            expected_properties={
                "min_length": 32,
                "region_change": True,
                "energy_increase": True,
            },
            genre=genre,
        )

    def _key_modulation(self, genre: str) -> ChallengeCase:
        """Challenge: modulate from C major to a related key."""
        states = []
        for bar in range(8):
            for beat in range(4):
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=2 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(0, 0),
                    harmonic_role=0,
                    melodic_head=60,
                    groove_token=0,
                ))

        return ChallengeCase(
            name="key_modulation",
            description="Modulate to a related key (G, F, or Am)",
            initial_context=BeatLevelSequence(
                states=states, bpm=120, key=0, is_major=True, genre=genre,
            ),
            expected_properties={
                "min_length": 32,
                "key_change": True,
            },
            genre=genre,
        )

    def _tension_arc(self, genre: str) -> ChallengeCase:
        """Challenge: produce a piece with a clear tension arc."""
        states = []
        for bar in range(4):
            for beat in range(4):
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=3 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(0, 0),
                    harmonic_role=0,
                    melodic_head=60,
                    groove_token=0,
                ))

        return ChallengeCase(
            name="tension_arc",
            description="Generate 32 bars with a clear rise-peak-fall tension arc",
            initial_context=BeatLevelSequence(states=states, bpm=120, genre=genre),
            expected_properties={
                "min_length": 128,
                "has_climax": True,
                "tension_range": 0.3,
            },
            genre=genre,
        )

    def _odd_meter(self, genre: str) -> ChallengeCase:
        """Challenge (prog rock): handle 7/8 time signature."""
        states = []
        for bar in range(4):
            for beat in range(7):  # 7/8
                states.append(BeatLevelState(
                    meter_token=4,  # 7/8
                    beat_position=beat,
                    boundary_level=2 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(0, 1),  # C minor
                    harmonic_role=0,
                    melodic_head=60,
                    groove_token=10,  # prog odd-meter
                ))

        return ChallengeCase(
            name="odd_meter_7_8",
            description="Continue in 7/8 time for 8 bars maintaining the meter",
            initial_context=BeatLevelSequence(states=states, bpm=140, genre=genre),
            expected_properties={
                "min_length": 56,
                "meter_consistent": True,
            },
            genre=genre,
        )

    def _ii_V_I_resolution(self, genre: str) -> ChallengeCase:
        """Challenge (jazz fusion): produce ii-V-I progressions."""
        states = []
        # Set up a ii-V context
        progressions = [
            (2, 4, 2),  # Dm7 (ii)
            (7, 2, 2),  # G7 (V)
        ]
        for bar, (root, quality, role) in enumerate(progressions):
            for beat in range(4):
                states.append(BeatLevelState(
                    meter_token=0,
                    beat_position=beat,
                    boundary_level=2 if bar == 0 and beat == 0 else 0,
                    region_label=0,
                    chord_label=encode_chord(root, quality),
                    harmonic_role=role,
                    melodic_head=62 + beat,
                    groove_token=8,  # jazz ride
                ))

        return ChallengeCase(
            name="ii_V_I_resolution",
            description="Resolve the ii-V to I and continue with jazz harmony",
            initial_context=BeatLevelSequence(states=states, bpm=130, genre=genre),
            expected_properties={
                "min_length": 16,
                "resolves_to_tonic": True,
            },
            genre=genre,
        )
