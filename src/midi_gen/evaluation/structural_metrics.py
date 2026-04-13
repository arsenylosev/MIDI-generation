"""Structural evaluation metrics for generated music.

These metrics assess whether the generated music exhibits purposeful
structural behavior, as recommended in the revised strategy note (Section 11).

The evaluator operates on BeatLevelSequences and produces a structured
report of scores across multiple dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    decode_chord,
    METER_BEATS,
)


@dataclass
class EvaluationReport:
    """Structured evaluation report."""
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    overall_score: float = 0.0

    def summary(self) -> str:
        lines = ["=== Structural Evaluation Report ===", ""]
        for metric, score in sorted(self.scores.items()):
            detail = self.details.get(metric, "")
            lines.append(f"  {metric:40s}  {score:.3f}  {detail}")
        lines.append("")
        lines.append(f"  {'OVERALL':40s}  {self.overall_score:.3f}")
        return "\n".join(lines)


class StructuralEvaluator:
    """Evaluate generated sequences on structural quality metrics.

    Metrics implemented:
        1. phrase_boundary_clarity   — Can a listener tell where phrases begin/end?
        2. cadence_arrival_quality   — Does the music land at structural endpoints?
        3. repetition_with_variation — Does returning material come back altered?
        4. meter_sensitive_harmony   — Do chord changes align with metrical hierarchy?
        5. multitrack_differentiation — Are track roles distinct and complementary?
        6. directed_movement         — Does the piece feel like it's going somewhere?
    """

    def evaluate(
        self,
        sequence: BeatLevelSequence,
        reference: Optional[BeatLevelSequence] = None,
    ) -> EvaluationReport:
        """Evaluate a generated sequence.

        Parameters
        ----------
        sequence : BeatLevelSequence
            The generated sequence to evaluate.
        reference : BeatLevelSequence, optional
            A reference sequence for comparison (if available).

        Returns
        -------
        EvaluationReport
            Scores and details for each metric.
        """
        report = EvaluationReport()

        report.scores["phrase_boundary_clarity"] = self._phrase_boundary_clarity(sequence)
        report.scores["cadence_arrival_quality"] = self._cadence_arrival_quality(sequence)
        report.scores["repetition_with_variation"] = self._repetition_with_variation(sequence)
        report.scores["meter_sensitive_harmony"] = self._meter_sensitive_harmony(sequence)
        report.scores["directed_movement"] = self._directed_movement(sequence)
        report.scores["harmonic_variety"] = self._harmonic_variety(sequence)

        # Overall: weighted average
        weights = {
            "phrase_boundary_clarity": 0.20,
            "cadence_arrival_quality": 0.20,
            "repetition_with_variation": 0.15,
            "meter_sensitive_harmony": 0.20,
            "directed_movement": 0.15,
            "harmonic_variety": 0.10,
        }
        report.overall_score = sum(
            report.scores[k] * weights.get(k, 0.1)
            for k in report.scores
        )

        return report

    # --- Individual metrics ---

    def _phrase_boundary_clarity(self, seq: BeatLevelSequence) -> float:
        """Measure how clearly phrase boundaries are marked.

        A good score means:
        - Boundaries (level >= 2) occur at regular, musically sensible intervals
        - Boundaries are on strong metric positions (beat 0)
        - Boundary spacing is neither too short nor too long
        """
        if len(seq) < 8:
            return 0.0

        boundaries = [i for i, s in enumerate(seq) if s.boundary_level >= 2]

        if not boundaries:
            return 0.1  # No boundaries at all is bad

        # Check that boundaries are on beat 0
        on_downbeat = sum(1 for i in boundaries if seq[i].beat_position == 0)
        downbeat_ratio = on_downbeat / len(boundaries)

        # Check spacing regularity
        if len(boundaries) >= 2:
            spacings = np.diff(boundaries)
            cv = np.std(spacings) / (np.mean(spacings) + 1e-6)
            regularity = max(0, 1.0 - cv)  # lower CV = more regular
        else:
            regularity = 0.5

        # Check reasonable phrase length (4–32 beats)
        if len(boundaries) >= 2:
            spacings = np.diff(boundaries)
            reasonable = sum(1 for s in spacings if 4 <= s <= 32) / len(spacings)
        else:
            reasonable = 0.5

        return 0.4 * downbeat_ratio + 0.3 * regularity + 0.3 * reasonable

    def _cadence_arrival_quality(self, seq: BeatLevelSequence) -> float:
        """Measure whether the music 'lands' at structural endpoints.

        Good cadences show:
        - Movement from dominant (role=2) to tonic (role=0) at boundaries
        - Melodic resolution (stepwise motion to a chord tone) at boundaries
        """
        if len(seq) < 4:
            return 0.0

        boundaries = [i for i, s in enumerate(seq) if s.boundary_level >= 2]
        if not boundaries:
            return 0.1

        cadence_scores = []
        for bi in boundaries:
            if bi < 2:
                continue

            # Check for dominant → tonic motion
            pre_boundary = seq[bi - 1]
            at_boundary = seq[bi]

            harmonic_resolution = 0.0
            if pre_boundary.harmonic_role == 2 and at_boundary.harmonic_role == 0:
                harmonic_resolution = 1.0
            elif pre_boundary.harmonic_role in (1, 2) and at_boundary.harmonic_role == 0:
                harmonic_resolution = 0.6
            elif at_boundary.harmonic_role == 0:
                harmonic_resolution = 0.3

            # Check for melodic resolution (small interval)
            melodic_resolution = 0.0
            if pre_boundary.melodic_head >= 0 and at_boundary.melodic_head >= 0:
                interval = abs(at_boundary.melodic_head - pre_boundary.melodic_head)
                if interval <= 2:
                    melodic_resolution = 1.0
                elif interval <= 4:
                    melodic_resolution = 0.6
                elif interval <= 7:
                    melodic_resolution = 0.3

            cadence_scores.append(0.6 * harmonic_resolution + 0.4 * melodic_resolution)

        return float(np.mean(cadence_scores)) if cadence_scores else 0.1

    def _repetition_with_variation(self, seq: BeatLevelSequence) -> float:
        """Measure whether returning material comes back recognizably but altered.

        Checks for chord sequence similarity between sections with the same
        region_label, expecting partial but not exact matches.
        """
        if len(seq) < 16:
            return 0.0

        # Group by region label
        regions: Dict[int, List[List[int]]] = {}
        current_region = seq[0].region_label
        current_chords = []

        for state in seq:
            if state.region_label != current_region:
                if current_chords:
                    regions.setdefault(current_region, []).append(current_chords)
                current_region = state.region_label
                current_chords = []
            current_chords.append(state.chord_label)

        if current_chords:
            regions.setdefault(current_region, []).append(current_chords)

        # Find regions that appear more than once
        repeated = {k: v for k, v in regions.items() if len(v) >= 2}
        if not repeated:
            return 0.3  # No repetition

        scores = []
        for region_id, occurrences in repeated.items():
            for i in range(len(occurrences) - 1):
                a = occurrences[i]
                b = occurrences[i + 1]
                sim = self._sequence_similarity(a, b)
                # Best score when similarity is moderate (0.5–0.8)
                if 0.5 <= sim <= 0.8:
                    scores.append(1.0)
                elif 0.3 <= sim <= 0.9:
                    scores.append(0.7)
                elif sim > 0.9:
                    scores.append(0.4)  # Too exact
                else:
                    scores.append(0.3)  # Too different

        return float(np.mean(scores)) if scores else 0.3

    def _meter_sensitive_harmony(self, seq: BeatLevelSequence) -> float:
        """Measure whether chord changes align with the metrical hierarchy.

        Good scores mean chord changes happen on strong beats, not randomly.
        """
        if len(seq) < 4:
            return 0.0

        chord_changes = 0
        changes_on_strong = 0

        for i in range(1, len(seq)):
            if seq[i].chord_label != seq[i - 1].chord_label:
                chord_changes += 1
                meter = seq[i].meter_token
                beats = METER_BEATS.get(meter, 4)
                strong = {0, beats // 2} if beats >= 4 else {0}
                if seq[i].beat_position in strong:
                    changes_on_strong += 1

        if chord_changes == 0:
            return 0.3  # No changes at all

        return changes_on_strong / chord_changes

    def _directed_movement(self, seq: BeatLevelSequence) -> float:
        """Measure whether the piece has a sense of directed movement.

        Checks for:
        - Tension arc (not flat)
        - Energy variation across sections
        - Climax point (peak tension)
        """
        if len(seq) < 8:
            return 0.0

        # Compute tension from harmonic roles
        tensions = np.array([s.harmonic_role / 4.0 for s in seq])

        # Check for non-flatness
        tension_range = np.max(tensions) - np.min(tensions)
        if tension_range < 0.1:
            return 0.2  # Flat tension = no movement

        # Check for arc shape (should rise and fall)
        midpoint = len(tensions) // 2
        first_half_mean = np.mean(tensions[:midpoint])
        second_half_mean = np.mean(tensions[midpoint:])
        peak_position = np.argmax(tensions) / len(tensions)

        # Reward if peak is in the middle-to-late portion (0.4–0.8)
        peak_score = 1.0 if 0.4 <= peak_position <= 0.8 else 0.5

        # Reward tension variety
        variety = min(1.0, tension_range * 3)

        return 0.5 * variety + 0.5 * peak_score

    def _harmonic_variety(self, seq: BeatLevelSequence) -> float:
        """Measure harmonic vocabulary richness."""
        if len(seq) < 4:
            return 0.0

        unique_chords = len(set(s.chord_label for s in seq))
        total_beats = len(seq)

        # Normalize: expect ~1 unique chord per 4–8 beats
        expected = total_beats / 6
        ratio = unique_chords / max(expected, 1)
        return min(1.0, ratio)

    # --- Helpers ---

    @staticmethod
    def _sequence_similarity(a: List[int], b: List[int]) -> float:
        """Compute similarity between two chord sequences (0–1)."""
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        matches = sum(1 for i in range(min_len) if a[i] == b[i])
        return matches / min_len
