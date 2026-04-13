"""Event normalizer: standardizes ingested data into a common representation.

Handles differences between gold, silver, and bronze data tiers:
    - Gold data is trusted at the note level
    - Silver data is trusted at the beat/bar level
    - Bronze data is only trusted at the bar/phrase level

The normalizer also handles transposition to a common key, tempo
normalization, and extraction of training examples for the scorer
and realizer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    encode_chord,
    decode_chord,
)
from midi_gen.corpus.ingestor import DataTier


class EventNormalizer:
    """Normalize and extract training data from ingested sequences.

    Parameters
    ----------
    transpose_to_c : bool
        Whether to transpose all sequences so the tonic is C.
    context_length : int
        Context window length for training example extraction.
    """

    def __init__(
        self,
        transpose_to_c: bool = True,
        context_length: int = 16,
    ):
        self.transpose_to_c = transpose_to_c
        self.context_length = context_length

    def normalize(
        self,
        sequence: BeatLevelSequence,
        tier: DataTier = DataTier.GOLD,
    ) -> BeatLevelSequence:
        """Normalize a sequence based on its data tier.

        Parameters
        ----------
        sequence : BeatLevelSequence
            The raw ingested sequence.
        tier : DataTier
            The data quality tier.

        Returns
        -------
        BeatLevelSequence
            The normalized sequence.
        """
        states = list(sequence.states)

        # Transpose to C if requested
        if self.transpose_to_c and sequence.key != 0:
            states = self._transpose(states, -sequence.key)

        # Tier-specific processing
        if tier == DataTier.BRONZE:
            # Bronze: smooth out note-level noise, keep only bar-level info
            states = self._smooth_bronze(states)
        elif tier == DataTier.SILVER:
            # Silver: keep beat-level info but smooth melodic heads
            states = self._smooth_silver(states)
        # Gold: keep as-is

        return BeatLevelSequence(
            states=states,
            bpm=sequence.bpm,
            key=0 if self.transpose_to_c else sequence.key,
            is_major=sequence.is_major,
            genre=sequence.genre,
            form_string=sequence.form_string,
        )

    def extract_scorer_examples(
        self,
        sequences: List[BeatLevelSequence],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract (context, next_state) pairs for scorer training.

        Parameters
        ----------
        sequences : list of BeatLevelSequence
            Normalized corpus sequences.

        Returns
        -------
        list of (context_matrix, next_state_vector) tuples
            Each context_matrix is (context_length, 8) and next_state is (8,).
        """
        examples = []
        ctx_len = self.context_length

        for seq in sequences:
            if len(seq) < ctx_len + 1:
                continue

            mat = seq.to_matrix()
            for i in range(ctx_len, len(seq)):
                context = mat[i - ctx_len:i]  # (ctx_len, 8)
                next_state = mat[i]  # (8,)
                examples.append((context, next_state))

        return examples

    def extract_realizer_examples(
        self,
        sequences: List[BeatLevelSequence],
        bars_per_example: int = 4,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract (structural_context, note_events) pairs for realizer training.

        Parameters
        ----------
        sequences : list of BeatLevelSequence
            Normalized corpus sequences.
        bars_per_example : int
            Number of bars per training example.

        Returns
        -------
        list of (context_matrix, bar_beats_matrix) tuples
        """
        examples = []

        for seq in sequences:
            bars = self._group_into_bars(seq)
            for start in range(0, len(bars) - bars_per_example + 1, bars_per_example):
                bar_group = bars[start:start + bars_per_example]
                all_beats = []
                for bar_beats in bar_group:
                    all_beats.extend(bar_beats)

                if all_beats:
                    mat = np.stack([s.to_vector() for s in all_beats])
                    # Context: first beat of each bar
                    ctx = np.stack([bar[0].to_vector() for bar in bar_group])
                    examples.append((ctx, mat))

        return examples

    # --- Internal helpers ---

    def _transpose(
        self, states: List[BeatLevelState], semitones: int,
    ) -> List[BeatLevelState]:
        """Transpose all pitch-related fields by the given number of semitones."""
        result = []
        for s in states:
            root, quality = decode_chord(s.chord_label)
            new_root = (root + semitones) % 12
            new_chord = encode_chord(new_root, quality)

            new_mel = s.melodic_head
            if new_mel >= 0:
                new_mel = max(0, min(127, new_mel + semitones))

            result.append(BeatLevelState(
                meter_token=s.meter_token,
                beat_position=s.beat_position,
                boundary_level=s.boundary_level,
                region_label=s.region_label,
                chord_label=new_chord,
                harmonic_role=s.harmonic_role,
                melodic_head=new_mel,
                groove_token=s.groove_token,
            ))
        return result

    def _smooth_bronze(self, states: List[BeatLevelState]) -> List[BeatLevelState]:
        """Smooth bronze-tier data: keep only bar-level information."""
        result = []
        for i, s in enumerate(states):
            # For bronze data, zero out melodic head (unreliable from transcription)
            # and keep only chord / boundary / groove info
            result.append(BeatLevelState(
                meter_token=s.meter_token,
                beat_position=s.beat_position,
                boundary_level=s.boundary_level,
                region_label=s.region_label,
                chord_label=s.chord_label,
                harmonic_role=s.harmonic_role,
                melodic_head=-1,  # unreliable in bronze data
                groove_token=s.groove_token,
            ))
        return result

    def _smooth_silver(self, states: List[BeatLevelState]) -> List[BeatLevelState]:
        """Smooth silver-tier data: keep beat-level info, smooth melodic heads."""
        result = []
        for i, s in enumerate(states):
            # Quantize melodic head to nearest scale degree
            mel = s.melodic_head
            if mel >= 0:
                # Snap to nearest chord tone
                root, quality = decode_chord(s.chord_label)
                mel = self._snap_to_chord_tone(mel, root, quality)

            result.append(BeatLevelState(
                meter_token=s.meter_token,
                beat_position=s.beat_position,
                boundary_level=s.boundary_level,
                region_label=s.region_label,
                chord_label=s.chord_label,
                harmonic_role=s.harmonic_role,
                melodic_head=mel,
                groove_token=s.groove_token,
            ))
        return result

    @staticmethod
    def _snap_to_chord_tone(pitch: int, root: int, quality: int) -> int:
        """Snap a pitch to the nearest chord tone."""
        intervals_map = {
            0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 4, 7, 10],
            3: [0, 4, 7, 11], 4: [0, 3, 7, 10], 5: [0, 3, 6],
            6: [0, 4, 8], 7: [0, 5, 7], 8: [0, 2, 7],
        }
        intervals = intervals_map.get(quality, [0, 4, 7])
        chord_pcs = set((root + iv) % 12 for iv in intervals)

        pc = pitch % 12
        if pc in chord_pcs:
            return pitch

        # Find nearest chord tone
        best_dist = 12
        best_pitch = pitch
        for cpc in chord_pcs:
            dist = min(abs(pc - cpc), 12 - abs(pc - cpc))
            if dist < best_dist:
                best_dist = dist
                offset = cpc - pc
                if abs(offset) > 6:
                    offset = offset - 12 if offset > 0 else offset + 12
                best_pitch = pitch + offset
        return max(0, min(127, best_pitch))

    @staticmethod
    def _group_into_bars(sequence: BeatLevelSequence) -> List[List[BeatLevelState]]:
        """Group beats into bars."""
        bars = []
        current_bar = []
        for state in sequence:
            if state.beat_position == 0 and current_bar:
                bars.append(current_bar)
                current_bar = []
            current_bar.append(state)
        if current_bar:
            bars.append(current_bar)
        return bars
