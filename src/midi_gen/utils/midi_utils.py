"""
MIDI Utilities for conversion between internal representations and MIDI files.

Provides functions for:
- Converting between piano-roll and note-matrix representations
- Reading and writing MIDI files via pretty_midi
- Converting between the internal MusicalState trajectory and
  the whole-song-gen compatible piano-roll format
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import os

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False


def trajectory_to_piano_roll(
    trajectory: list,
    steps_per_beat: int = 4,
    pitch_range: Tuple[int, int] = (0, 128)
) -> np.ndarray:
    """
    Convert a list of MusicalState objects to a piano-roll representation.

    Returns a (2, T, 128) array with onset and sustain channels,
    compatible with the whole-song-gen format.
    """
    num_steps = len(trajectory)
    piano_roll = np.zeros((2, num_steps, 128))

    prev_pitches = set()
    for t, state in enumerate(trajectory):
        active_pitches = set()
        for p in range(pitch_range[0], pitch_range[1]):
            if state.pitch_vector[p] > 0.5:
                active_pitches.add(p)

        for p in active_pitches:
            if p not in prev_pitches:
                piano_roll[0, t, p] = 1.0  # onset
            else:
                piano_roll[1, t, p] = 1.0  # sustain

        prev_pitches = active_pitches

    return piano_roll


def piano_roll_to_note_list(
    piano_roll: np.ndarray,
    bpm: float = 120.0,
    steps_per_beat: int = 4
) -> List[Dict]:
    """
    Convert a (2, T, 128) piano-roll to a list of note dictionaries.

    Each note has: pitch, start_time, end_time, velocity.
    """
    assert piano_roll.shape[0] == 2 and piano_roll.shape[2] == 128

    step_duration = 60.0 / bpm / steps_per_beat
    notes = []
    num_steps = piano_roll.shape[1]

    for p in range(128):
        in_note = False
        note_start = 0

        for t in range(num_steps):
            is_onset = piano_roll[0, t, p] > 0.5
            is_sustain = piano_roll[1, t, p] > 0.5

            if is_onset:
                if in_note:
                    notes.append({
                        "pitch": p,
                        "start_time": note_start * step_duration,
                        "end_time": t * step_duration,
                        "velocity": 100
                    })
                note_start = t
                in_note = True
            elif not is_sustain and in_note:
                notes.append({
                    "pitch": p,
                    "start_time": note_start * step_duration,
                    "end_time": t * step_duration,
                    "velocity": 100
                })
                in_note = False

        if in_note:
            notes.append({
                "pitch": p,
                "start_time": note_start * step_duration,
                "end_time": num_steps * step_duration,
                "velocity": 100
            })

    notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return notes


def note_list_to_midi(
    notes: List[Dict],
    bpm: float = 120.0,
    instrument_name: str = "Acoustic Grand Piano",
    program: int = 0
) -> "pretty_midi.PrettyMIDI":
    """Convert a note list to a PrettyMIDI object."""
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi is required. Install with: pip install pretty_midi")

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=program, name=instrument_name)

    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=note.get("velocity", 100),
            pitch=note["pitch"],
            start=note["start_time"],
            end=note["end_time"]
        )
        instrument.notes.append(midi_note)

    midi.instruments.append(instrument)
    return midi


def save_midi(midi_obj, filepath: str):
    """Save a PrettyMIDI object to a file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    midi_obj.write(filepath)


def load_midi(filepath: str) -> "pretty_midi.PrettyMIDI":
    """Load a MIDI file and return a PrettyMIDI object."""
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi is required.")
    return pretty_midi.PrettyMIDI(filepath)


def midi_to_piano_roll(
    midi_obj,
    steps_per_beat: int = 4,
    bpm: Optional[float] = None
) -> np.ndarray:
    """
    Convert a PrettyMIDI object to a (2, T, 128) piano-roll.

    If bpm is not specified, uses the tempo from the MIDI file.
    """
    if bpm is None:
        tempos = midi_obj.get_tempo_changes()
        bpm = tempos[1][0] if len(tempos[1]) > 0 else 120.0

    step_duration = 60.0 / bpm / steps_per_beat
    end_time = midi_obj.get_end_time()
    num_steps = int(np.ceil(end_time / step_duration))

    piano_roll = np.zeros((2, num_steps, 128))

    for instrument in midi_obj.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start_step = int(round(note.start / step_duration))
            end_step = int(round(note.end / step_duration))
            if 0 <= note.pitch < 128 and start_step < num_steps:
                piano_roll[0, start_step, note.pitch] = 1.0  # onset
                for t in range(start_step + 1, min(end_step, num_steps)):
                    piano_roll[1, t, note.pitch] = 1.0  # sustain

    return piano_roll


def create_whole_song_gen_input(
    piano_roll: np.ndarray,
    phrase_structure: List[Dict],
    key: int = 0,
    is_major: bool = True
) -> Dict:
    """
    Package piano-roll and structure info into a format compatible
    with the whole-song-gen pipeline.

    This creates the 'form' level input that the cascaded diffusion
    model expects.
    """
    # Build phrase string (e.g., "i4A4A4B8b4A4B8o4")
    type_map = {
        "intro": "i", "verse": "A", "chorus": "B",
        "bridge": "b", "solo": "S", "outro": "o",
        "theme_a": "A", "theme_b": "B", "development": "C",
        "recapitulation": "A", "coda": "o",
        "head_in": "A", "solo_section": "S", "interlude": "b",
        "trading_fours": "T", "head_out": "A",
        "theme_a_reprise": "A", "solo_1": "S", "solo_2": "S",
    }

    phrase_string = ""
    for phrase in phrase_structure:
        ptype = type_map.get(phrase["name"], phrase.get("type", "A"))
        phrase_string += f"{ptype}{phrase['lgth']}"

    return {
        "piano_roll": piano_roll,
        "phrase_string": phrase_string,
        "key": key,
        "is_major": is_major,
        "phrase_structure": phrase_structure
    }


def generate_form_string(phrases: List[Dict]) -> str:
    """Generate a whole-song-gen compatible form string from phrase list."""
    type_map = {
        "intro": "i", "verse": "A", "chorus": "B",
        "bridge": "b", "solo": "S", "outro": "o",
        "theme_a": "A", "theme_b": "B", "development": "C",
        "recapitulation": "A", "coda": "o",
        "head_in": "A", "solo_section": "S", "interlude": "b",
        "trading_fours": "T", "head_out": "A",
        "theme_a_reprise": "A", "solo_1": "S", "solo_2": "S",
    }
    parts = []
    for phrase in phrases:
        ptype = type_map.get(phrase["name"], phrase.get("type", "A"))
        parts.append(f"{ptype}{phrase['lgth']}")
    return "".join(parts)
