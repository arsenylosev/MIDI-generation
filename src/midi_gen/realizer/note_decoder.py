"""Note decoder: converts bar-level texture codes into concrete MIDI events.

For each bar, the decoder takes the BarTextureCode and the beat-level
structural context (chords, melody, groove) and produces note events
for each track role.

This is a rule-based decoder that implements common arrangement patterns.
It can later be replaced by a learned decoder (Phase 2 of the build order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence, decode_chord, METER_BEATS
from midi_gen.schema.bar_texture import BarTextureCode, TrackRole


@dataclass
class NoteEvent:
    """A single MIDI note event."""
    track: TrackRole
    pitch: int
    velocity: int
    start_beat: float   # in beats from bar start
    duration: float     # in beats
    channel: int = 0


@dataclass
class BarEvents:
    """All note events for one bar, grouped by track."""
    bar_index: int = 0
    events: List[NoteEvent] = field(default_factory=list)

    def by_track(self) -> Dict[TrackRole, List[NoteEvent]]:
        result = {role: [] for role in TrackRole}
        for e in self.events:
            result[e.track].append(e)
        return result


class NoteDecoder:
    """Decode texture codes into concrete note events.

    Parameters
    ----------
    genre : str
        Target genre (affects pattern details).
    velocity_base : int
        Base velocity for notes (0–127).
    """

    def __init__(self, genre: str = "prog_rock", velocity_base: int = 80):
        self.genre = genre
        self.velocity_base = velocity_base

    def decode_bar(
        self,
        texture: BarTextureCode,
        bar_beats: List[BeatLevelState],
        beats_per_bar: int = 4,
    ) -> BarEvents:
        """Decode one bar of texture codes into note events.

        Parameters
        ----------
        texture : BarTextureCode
            The texture codes for this bar.
        bar_beats : list of BeatLevelState
            The beat-level states for this bar.
        beats_per_bar : int
            Number of beats in this bar.

        Returns
        -------
        BarEvents
            Concrete note events for all tracks.
        """
        events = BarEvents(bar_index=texture.bar_index)
        vel = int(self.velocity_base * texture.energy)
        vel = max(30, min(127, vel))

        # Get chord info from first beat
        if bar_beats:
            root, quality = decode_chord(bar_beats[0].chord_label)
        else:
            root, quality = 0, 0

        chord_pitches = self._chord_pitches(root, quality)

        # Decode each track
        events.events.extend(self._decode_drums(texture.drums, beats_per_bar, vel))
        events.events.extend(self._decode_bass(texture.bass, root, beats_per_bar, vel))
        events.events.extend(self._decode_comping(texture.comping, chord_pitches, beats_per_bar, vel))
        events.events.extend(self._decode_lead(texture.lead, bar_beats, beats_per_bar, vel))

        return events

    def decode_sequence(
        self,
        plan: "ArrangementPlan",
        sequence: BeatLevelSequence,
    ) -> List[BarEvents]:
        """Decode an entire arrangement plan into bar events."""
        from midi_gen.realizer.texture_planner import TexturePlanner

        bars_beats = self._group_beats_into_bars(sequence)
        result = []

        for i, texture in enumerate(plan.bars):
            if i < len(bars_beats):
                bar_beats = bars_beats[i]
                bpb = METER_BEATS.get(bar_beats[0].meter_token, 4) if bar_beats else 4
            else:
                bar_beats = []
                bpb = 4

            result.append(self.decode_bar(texture, bar_beats, bpb))

        return result

    # --- Track decoders ---

    def _decode_drums(
        self, groove_code: int, beats_per_bar: int, vel: int,
    ) -> List[NoteEvent]:
        """Generate drum events based on groove family code."""
        events = []

        if groove_code == 15:  # tacet
            return events

        # GM drum map: kick=36, snare=38, hihat=42, ride=51, crash=49
        KICK, SNARE, HIHAT, RIDE, CRASH = 36, 38, 42, 51, 49

        # Basic patterns by groove family
        if groove_code in (0, 1):  # straight 8th / 16th
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 10, b, 0.25))
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 20, b + 0.5, 0.25))
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel - 5, beats_per_bar / 2, 0.5))
            events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 1, 0.5))
            if beats_per_bar >= 4:
                events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 3, 0.5))

        elif groove_code == 2:  # shuffle
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 10, b, 0.33))
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 25, b + 0.67, 0.17))
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 1, 0.5))
            if beats_per_bar >= 4:
                events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 3, 0.5))

        elif groove_code == 3:  # half-time
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            if beats_per_bar >= 3:
                events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 2, 0.5))
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 15, b, 0.5))

        elif groove_code in (4, 5):  # double-time / 6/8 feel
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 10, b, 0.17))
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 20, b + 0.33, 0.17))
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 20, b + 0.67, 0.17))
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 1, 0.5))

        elif groove_code in (8, 9):  # jazz ride / brush
            instrument = RIDE if groove_code == 8 else HIHAT
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, instrument, vel - 10, b, 0.33))
                events.append(NoteEvent(TrackRole.DRUMS, instrument, vel - 25, b + 0.67, 0.17))
            # Jazz kick: light, on beats 1 and 3
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel - 20, 0, 0.5))
            if beats_per_bar >= 4:
                events.append(NoteEvent(TrackRole.DRUMS, KICK, vel - 25, 2, 0.5))

        elif groove_code == 10:  # prog odd-meter
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 10, b, 0.25))
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, min(2, beats_per_bar - 1), 0.5))
            if beats_per_bar >= 5:
                events.append(NoteEvent(TrackRole.DRUMS, KICK, vel - 5, 3, 0.5))

        else:  # default simple pattern
            events.append(NoteEvent(TrackRole.DRUMS, KICK, vel, 0, 0.5))
            if beats_per_bar >= 2:
                events.append(NoteEvent(TrackRole.DRUMS, SNARE, vel, 1, 0.5))
            for b in range(beats_per_bar):
                events.append(NoteEvent(TrackRole.DRUMS, HIHAT, vel - 15, b, 0.5))

        # Set channel 9 for drums
        for e in events:
            e.channel = 9

        return events

    def _decode_bass(
        self, bass_code: int, root: int, beats_per_bar: int, vel: int,
    ) -> List[NoteEvent]:
        """Generate bass events based on bass motion code."""
        events = []
        base_pitch = 36 + root  # Bass octave (C2 range)

        if bass_code == 15:  # tacet
            return events

        if bass_code == 0:  # root only
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel, 0, beats_per_bar))

        elif bass_code == 1:  # root-fifth
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel, 0, beats_per_bar / 2))
            events.append(NoteEvent(TrackRole.BASS, base_pitch + 7, vel - 5, beats_per_bar / 2, beats_per_bar / 2))

        elif bass_code == 2:  # walking
            intervals = [0, 5, 7, 12, 7, 5, 0, -5]
            for b in range(beats_per_bar):
                iv = intervals[b % len(intervals)]
                p = max(28, min(60, base_pitch + iv))
                events.append(NoteEvent(TrackRole.BASS, p, vel - 5, b, 0.9))

        elif bass_code == 3:  # pedal
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel, 0, beats_per_bar))

        elif bass_code == 4:  # riff
            riff = [0, 0, 3, 5, 7, 5, 3, 0]
            step = beats_per_bar / min(beats_per_bar, 4)
            for i in range(min(beats_per_bar, 4)):
                iv = riff[i % len(riff)]
                p = max(28, min(60, base_pitch + iv))
                events.append(NoteEvent(TrackRole.BASS, p, vel, i * step, step * 0.9))

        elif bass_code == 9:  # syncopated
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel, 0, 0.75))
            events.append(NoteEvent(TrackRole.BASS, base_pitch + 7, vel - 5, 1.5, 0.75))
            if beats_per_bar >= 4:
                events.append(NoteEvent(TrackRole.BASS, base_pitch + 5, vel - 5, 2.5, 0.75))

        elif bass_code == 14:  # drone
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel - 10, 0, beats_per_bar))

        else:  # default: root on downbeats
            events.append(NoteEvent(TrackRole.BASS, base_pitch, vel, 0, beats_per_bar / 2))
            if beats_per_bar >= 4:
                events.append(NoteEvent(TrackRole.BASS, base_pitch, vel - 10, beats_per_bar / 2, beats_per_bar / 2))

        return events

    def _decode_comping(
        self, comp_code: int, chord_pitches: List[int], beats_per_bar: int, vel: int,
    ) -> List[NoteEvent]:
        """Generate comping events based on comping style code."""
        events = []
        # Shift chord to mid range (C4 area)
        pitches = [p + 48 for p in chord_pitches if 0 <= p + 48 <= 127]

        if comp_code == 15:  # tacet
            return events

        if comp_code == 0:  # block chords
            for p in pitches:
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 10, 0, beats_per_bar))

        elif comp_code == 1:  # arpeggiated
            step = beats_per_bar / max(len(pitches), 1)
            for i, p in enumerate(pitches):
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 10, i * step, step * 0.9))

        elif comp_code == 2:  # rhythmic stabs
            for b in [0, 1.5, 3] if beats_per_bar >= 4 else [0, 1]:
                if b < beats_per_bar:
                    for p in pitches:
                        events.append(NoteEvent(TrackRole.COMPING, p, vel, b, 0.25))

        elif comp_code == 3:  # sustained pads
            for p in pitches:
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 20, 0, beats_per_bar))

        elif comp_code == 4:  # sparse hits
            for p in pitches:
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 5, 0, 0.5))

        elif comp_code == 5:  # dense voicings
            for b in range(beats_per_bar):
                for p in pitches:
                    events.append(NoteEvent(TrackRole.COMPING, p, vel - 10, b, 0.9))

        elif comp_code == 7:  # shell voicings (jazz)
            shells = pitches[:2] if len(pitches) >= 2 else pitches
            for p in shells:
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 15, 0, beats_per_bar * 0.8))

        else:  # default: whole-bar chord
            for p in pitches:
                events.append(NoteEvent(TrackRole.COMPING, p, vel - 15, 0, beats_per_bar))

        return events

    def _decode_lead(
        self, lead_code: int, bar_beats: List[BeatLevelState], beats_per_bar: int, vel: int,
    ) -> List[NoteEvent]:
        """Generate lead melody events from beat-level melodic heads."""
        events = []

        if lead_code in (12, 13):  # silence / fade
            return events

        for i, state in enumerate(bar_beats):
            if state.melodic_head < 0:
                continue

            pitch = state.melodic_head
            # Determine duration: hold until next different pitch or bar end
            dur = 1.0
            for j in range(i + 1, len(bar_beats)):
                if bar_beats[j].melodic_head != pitch:
                    break
                dur += 1.0

            dur = min(dur, beats_per_bar - i)
            events.append(NoteEvent(TrackRole.LEAD, pitch, vel, float(i), dur * 0.95))

        return events

    # --- Helpers ---

    def _chord_pitches(self, root: int, quality: int) -> List[int]:
        """Get chord intervals relative to root."""
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
        return [root + iv for iv in intervals]

    def _group_beats_into_bars(
        self, sequence: BeatLevelSequence,
    ) -> List[List[BeatLevelState]]:
        """Group beats into bars based on beat_position resets."""
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
