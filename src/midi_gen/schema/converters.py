"""Converters between the native beat-level representation and interchange formats.

The piano-roll (2, T, 128) tensor is kept as a *compatibility layer* for
whole-song-gen.  These converters translate between the native BeatLevelSequence
and that interchange format, as well as standard MIDI files.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    decode_chord,
    METER_BEATS,
    INV_METER_MAP,
)


def beat_sequence_to_piano_roll(
    seq: BeatLevelSequence,
    steps_per_beat: int = 4,
) -> np.ndarray:
    """Convert a BeatLevelSequence to a (2, T, 128) onset/sustain piano-roll.

    This is the interchange format expected by the whole-song-gen adapter.
    Each beat in the sequence is expanded to ``steps_per_beat`` time steps.
    Melodic heads are placed as onsets; chord tones are added as sustained
    accompaniment based on the chord_label field.

    Parameters
    ----------
    seq : BeatLevelSequence
        The native structural trajectory.
    steps_per_beat : int
        Number of 16th-note steps per beat (default 4).

    Returns
    -------
    np.ndarray
        Shape ``(2, T, 128)`` where T = len(seq) * steps_per_beat.
    """
    n_beats = len(seq)
    T = n_beats * steps_per_beat
    piano_roll = np.zeros((2, T, 128), dtype=np.float32)

    for i, state in enumerate(seq):
        t_start = i * steps_per_beat

        # Place melodic head as onset on the first step of the beat
        if 0 <= state.melodic_head <= 127:
            piano_roll[0, t_start, state.melodic_head] = 1.0  # onset
            for dt in range(1, steps_per_beat):
                piano_roll[1, t_start + dt, state.melodic_head] = 1.0  # sustain

        # Add chord tones as sustained accompaniment
        root, quality = decode_chord(state.chord_label)
        chord_pitches = _chord_to_pitches(root, quality, octave=4)
        for p in chord_pitches:
            if 0 <= p <= 127:
                piano_roll[0, t_start, p] = 1.0
                for dt in range(1, steps_per_beat):
                    piano_roll[1, t_start + dt, p] = 1.0

    return piano_roll


def piano_roll_to_beat_sequence(
    piano_roll: np.ndarray,
    steps_per_beat: int = 4,
    bpm: float = 120.0,
    key: int = 0,
    is_major: bool = True,
    genre: str = "prog_rock",
) -> BeatLevelSequence:
    """Convert a (2, T, 128) piano-roll back to a BeatLevelSequence.

    This is a *lossy* reverse mapping — the piano-roll does not carry the
    full structural information, so many fields are set to defaults.  It is
    useful for importing external piano-roll data into the native format for
    further processing.
    """
    T = piano_roll.shape[1]
    n_beats = T // steps_per_beat
    states = []

    for i in range(n_beats):
        t_start = i * steps_per_beat
        onset_slice = piano_roll[0, t_start, :]

        # Find the highest-pitched onset as the melodic head
        active = np.where(onset_slice > 0.5)[0]
        melodic_head = int(active[-1]) if len(active) > 0 else -1

        states.append(BeatLevelState(
            meter_token=0,
            beat_position=i % 4,
            boundary_level=0,
            region_label=0,
            chord_label=0,
            harmonic_role=0,
            melodic_head=melodic_head,
            groove_token=0,
        ))

    return BeatLevelSequence(
        states=states,
        bpm=bpm,
        key=key,
        is_major=is_major,
        genre=genre,
    )


def beat_sequence_to_midi(
    seq: BeatLevelSequence,
    output_path: Optional[str] = None,
) -> Optional["pretty_midi.PrettyMIDI"]:
    """Convert a BeatLevelSequence to a PrettyMIDI object.

    Creates two tracks: a melody track (from melodic_head) and a chord
    track (from chord_label).

    Parameters
    ----------
    seq : BeatLevelSequence
        The native structural trajectory.
    output_path : str, optional
        If given, writes the MIDI file to this path.

    Returns
    -------
    pretty_midi.PrettyMIDI or None
        The MIDI object, or None if pretty_midi is not installed.
    """
    if pretty_midi is None:
        return None

    pm = pretty_midi.PrettyMIDI(initial_tempo=seq.bpm)
    seconds_per_beat = 60.0 / seq.bpm

    # Melody track
    melody_inst = pretty_midi.Instrument(program=0, name="Melody")
    # Chord track
    chord_inst = pretty_midi.Instrument(program=4, name="Chords")

    i = 0
    while i < len(seq):
        state = seq[i]
        t_start = i * seconds_per_beat

        # Melody note (merge consecutive sustains of the same pitch)
        if 0 <= state.melodic_head <= 127:
            pitch = state.melodic_head
            j = i + 1
            while j < len(seq) and seq[j].melodic_head == pitch and seq[j].boundary_level == 0:
                j += 1
            t_end = j * seconds_per_beat
            melody_inst.notes.append(
                pretty_midi.Note(velocity=90, pitch=pitch, start=t_start, end=t_end)
            )

        # Chord notes (one beat duration)
        root, quality = decode_chord(state.chord_label)
        chord_pitches = _chord_to_pitches(root, quality, octave=3)
        t_end_chord = (i + 1) * seconds_per_beat
        for p in chord_pitches:
            if 0 <= p <= 127:
                chord_inst.notes.append(
                    pretty_midi.Note(velocity=60, pitch=p, start=t_start, end=t_end_chord)
                )

        i += 1

    pm.instruments.append(melody_inst)
    pm.instruments.append(chord_inst)

    if output_path:
        pm.write(output_path)

    return pm


# ---------------------------------------------------------------------------
# Aliases for convenience
# ---------------------------------------------------------------------------

# Short aliases used by the pipeline and tests
sequence_to_piano_roll = beat_sequence_to_piano_roll
piano_roll_to_sequence = piano_roll_to_beat_sequence
sequence_to_midi = beat_sequence_to_midi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chord_to_pitches(root: int, quality: int, octave: int = 4) -> list[int]:
    """Convert a chord (root, quality) to a list of MIDI pitches."""
    base = 12 * octave + root
    intervals_map = {
        0: [0, 4, 7],       # maj
        1: [0, 3, 7],       # min
        2: [0, 4, 7, 10],   # dom7
        3: [0, 4, 7, 11],   # maj7
        4: [0, 3, 7, 10],   # min7
        5: [0, 3, 6],       # dim
        6: [0, 4, 8],       # aug
        7: [0, 5, 7],       # sus4
        8: [0, 2, 7],       # sus2
    }
    intervals = intervals_map.get(quality, [0, 4, 7])
    return [base + iv for iv in intervals]
