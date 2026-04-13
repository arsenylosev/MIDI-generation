"""Multitrack realizer: the top-level module that converts a structural
trajectory into a full multitrack MIDI arrangement.

Orchestrates the two-stage process:
    1. TexturePlanner → assigns BarTextureCodes
    2. NoteDecoder → decodes into concrete note events

Then assembles the note events into a PrettyMIDI object with separate
tracks for drums, bass, comping, lead, and aux.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from midi_gen.schema.beat_state import BeatLevelSequence, METER_BEATS
from midi_gen.schema.bar_texture import ArrangementPlan, TrackRole
from midi_gen.realizer.texture_planner import TexturePlanner
from midi_gen.realizer.note_decoder import NoteDecoder, NoteEvent, BarEvents


# GM program numbers for each track role
TRACK_PROGRAMS = {
    TrackRole.DRUMS: 0,     # channel 9 (no program needed)
    TrackRole.BASS: 33,     # Electric Bass (finger)
    TrackRole.COMPING: 4,   # Electric Piano 1
    TrackRole.LEAD: 0,      # Acoustic Grand Piano
    TrackRole.AUX: 89,      # Pad 2 (warm)
}

TRACK_NAMES = {
    TrackRole.DRUMS: "Drums",
    TrackRole.BASS: "Bass",
    TrackRole.COMPING: "Comping",
    TrackRole.LEAD: "Lead",
    TrackRole.AUX: "Aux/Pad",
}


class MultitrackRealizer:
    """Convert a beat-level structural trajectory into multitrack MIDI.

    Parameters
    ----------
    genre : str
        Target genre.
    velocity_base : int
        Base velocity for generated notes.
    """

    def __init__(self, genre: str = "prog_rock", velocity_base: int = 80):
        self.genre = genre
        self.planner = TexturePlanner(genre=genre)
        self.decoder = NoteDecoder(genre=genre, velocity_base=velocity_base)

    def realize(
        self,
        sequence: BeatLevelSequence,
        tension_curve: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> Optional["pretty_midi.PrettyMIDI"]:
        """Generate a multitrack MIDI arrangement.

        Parameters
        ----------
        sequence : BeatLevelSequence
            The beat-level structural trajectory from the planner.
        tension_curve : np.ndarray, optional
            Per-beat tension values (0–1).
        output_path : str, optional
            If given, writes the MIDI file to this path.

        Returns
        -------
        pretty_midi.PrettyMIDI or None
            The multitrack MIDI object, or None if pretty_midi is unavailable.
        """
        # Stage 1: Texture planning
        arrangement = self.planner.plan(sequence, tension_curve)

        # Stage 2: Note decoding
        bar_events_list = self.decoder.decode_sequence(arrangement, sequence)

        # Stage 3: Assemble into PrettyMIDI
        pm = self._assemble_midi(sequence, bar_events_list)

        if pm is not None and output_path:
            pm.write(output_path)

        return pm

    def realize_to_arrangement(
        self,
        sequence: BeatLevelSequence,
        tension_curve: Optional[np.ndarray] = None,
    ) -> ArrangementPlan:
        """Return just the arrangement plan (texture codes) without MIDI."""
        return self.planner.plan(sequence, tension_curve)

    def _assemble_midi(
        self,
        sequence: BeatLevelSequence,
        bar_events_list: List[BarEvents],
    ) -> Optional["pretty_midi.PrettyMIDI"]:
        """Assemble bar events into a PrettyMIDI object."""
        if pretty_midi is None:
            return None

        pm = pretty_midi.PrettyMIDI(initial_tempo=sequence.bpm)
        seconds_per_beat = 60.0 / sequence.bpm

        # Create instrument tracks
        instruments = {}
        for role in TrackRole:
            is_drum = (role == TrackRole.DRUMS)
            program = TRACK_PROGRAMS[role]
            inst = pretty_midi.Instrument(
                program=program,
                is_drum=is_drum,
                name=TRACK_NAMES[role],
            )
            instruments[role] = inst

        # Compute bar start times in seconds
        bar_start_times = self._compute_bar_starts(sequence, seconds_per_beat)

        # Add notes from each bar
        for bar_idx, bar_events in enumerate(bar_events_list):
            if bar_idx >= len(bar_start_times):
                break

            bar_start_sec = bar_start_times[bar_idx]

            for event in bar_events.events:
                start = bar_start_sec + event.start_beat * seconds_per_beat
                end = start + event.duration * seconds_per_beat
                end = max(end, start + 0.01)  # minimum duration

                note = pretty_midi.Note(
                    velocity=max(1, min(127, event.velocity)),
                    pitch=max(0, min(127, event.pitch)),
                    start=start,
                    end=end,
                )
                instruments[event.track].notes.append(note)

        # Add instruments to MIDI object
        for role in TrackRole:
            if instruments[role].notes:
                pm.instruments.append(instruments[role])

        return pm

    def _compute_bar_starts(
        self,
        sequence: BeatLevelSequence,
        seconds_per_beat: float,
    ) -> List[float]:
        """Compute the start time of each bar in seconds."""
        bar_starts = [0.0]
        beat_count = 0

        for i, state in enumerate(sequence):
            if i > 0 and state.beat_position == 0:
                bar_starts.append(beat_count * seconds_per_beat)
            beat_count += 1

        return bar_starts
