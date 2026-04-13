"""Guide-audio renderer: produces per-stem guide channels for audio models.

Implements the guide-audio conditioning strategy from the revised strategy
note (Sections 7–8):

    Instead of rendering a single mixed audio file, produce separate guide
    channels aligned to phrase boundaries:

        1. Lead / melody guide
        2. Drum / groove guide
        3. Harmony / bass guide
        4. (Optional) Texture / pad guide

    These guides are intentionally rough — just clear enough to communicate
    pitch, rhythm, and arrangement structure.  They are designed to be fed
    into contemporary music audio models (e.g., MusicGen, Stable Audio) as
    conditioning signals.

The phrase structure from the GTTM planner provides the natural segmentation
unit for guide rendering.
"""

from __future__ import annotations

import os
import struct
import wave
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from midi_gen.schema.beat_state import BeatLevelSequence
from midi_gen.schema.bar_texture import TrackRole


@dataclass
class GuideChannel:
    """A single guide audio channel for one track role."""
    role: TrackRole
    audio: np.ndarray       # mono audio samples
    sample_rate: int = 44100
    name: str = ""

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass
class PhraseGuide:
    """Guide channels for a single phrase segment."""
    phrase_index: int
    start_sec: float
    end_sec: float
    channels: Dict[str, GuideChannel] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class GuideBundle:
    """Complete set of guide channels for the whole piece."""
    phrases: List[PhraseGuide] = field(default_factory=list)
    full_channels: Dict[str, GuideChannel] = field(default_factory=dict)
    bpm: float = 120.0
    genre: str = "prog_rock"

    @property
    def duration_sec(self) -> float:
        if self.full_channels:
            return max(ch.duration_sec for ch in self.full_channels.values())
        return 0.0


class GuideAudioRenderer:
    """Render multitrack MIDI into separate guide audio channels.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate.
    soundfont_path : str, optional
        Path to a SoundFont file for FluidSynth rendering.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        soundfont_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.soundfont_path = soundfont_path

    def render_guides(
        self,
        midi_path: str,
        sequence: Optional[BeatLevelSequence] = None,
        output_dir: Optional[str] = None,
    ) -> GuideBundle:
        """Render a multitrack MIDI file into guide channels.

        Parameters
        ----------
        midi_path : str
            Path to the multitrack MIDI file (from the realizer).
        sequence : BeatLevelSequence, optional
            The structural trajectory (for phrase boundary detection).
        output_dir : str, optional
            If given, saves individual WAV files per channel and phrase.

        Returns
        -------
        GuideBundle
            The complete set of guide channels.
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for guide rendering")

        pm = pretty_midi.PrettyMIDI(midi_path)
        duration = pm.get_end_time() + 0.5

        # Render each instrument track separately
        full_channels = {}
        for inst in pm.instruments:
            role_name = self._classify_instrument(inst)
            audio = self._render_instrument(inst, duration)
            full_channels[role_name] = GuideChannel(
                role=self._name_to_role(role_name),
                audio=audio,
                sample_rate=self.sample_rate,
                name=role_name,
            )

        # Detect phrase boundaries
        phrases = self._segment_phrases(sequence, duration)

        # Create phrase-level guide segments
        phrase_guides = []
        for i, (start, end) in enumerate(phrases):
            pg = PhraseGuide(phrase_index=i, start_sec=start, end_sec=end)
            for name, channel in full_channels.items():
                s_sample = int(start * self.sample_rate)
                e_sample = int(end * self.sample_rate)
                e_sample = min(e_sample, len(channel.audio))
                if s_sample < e_sample:
                    segment = channel.audio[s_sample:e_sample]
                else:
                    segment = np.zeros(int((end - start) * self.sample_rate))
                pg.channels[name] = GuideChannel(
                    role=channel.role,
                    audio=segment,
                    sample_rate=self.sample_rate,
                    name=f"{name}_phrase{i}",
                )
            phrase_guides.append(pg)

        bundle = GuideBundle(
            phrases=phrase_guides,
            full_channels=full_channels,
            bpm=pm.estimate_tempo() if sequence is None else sequence.bpm,
            genre=sequence.genre if sequence else "prog_rock",
        )

        # Save to disk if requested
        if output_dir:
            self._save_bundle(bundle, output_dir)

        return bundle

    def render_stem_midi(
        self,
        midi_path: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """Split a multitrack MIDI into separate per-stem MIDI files.

        This is useful for feeding individual stems to different audio
        models or for debugging the arrangement.

        Returns
        -------
        dict
            Mapping from stem name to output MIDI file path.
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required")

        pm = pretty_midi.PrettyMIDI(midi_path)
        os.makedirs(output_dir, exist_ok=True)
        stems = {}

        for inst in pm.instruments:
            role_name = self._classify_instrument(inst)
            stem_pm = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
            stem_pm.instruments.append(inst)
            stem_path = os.path.join(output_dir, f"{role_name}.mid")
            stem_pm.write(stem_path)
            stems[role_name] = stem_path

        return stems

    # --- Internal methods ---

    def _render_instrument(
        self,
        instrument: "pretty_midi.Instrument",
        duration: float,
    ) -> np.ndarray:
        """Render a single instrument to audio using additive synthesis."""
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        for note in instrument.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            s = int(note.start * self.sample_rate)
            e = min(int(note.end * self.sample_rate), num_samples)
            if s >= e:
                continue

            t = np.arange(e - s) / self.sample_rate
            vel = note.velocity / 127.0

            if instrument.is_drum:
                # Drum: short noise burst
                wave_data = vel * 0.3 * np.random.randn(e - s)
                env_len = min(len(wave_data), int(0.1 * self.sample_rate))
                envelope = np.ones(len(wave_data))
                if env_len > 0:
                    envelope[env_len:] = 0
                    envelope[:env_len] = np.linspace(1, 0, env_len)
                wave_data *= envelope
            else:
                # Pitched: additive synthesis with ADSR
                wave_data = np.zeros_like(t)
                harmonics = [(1.0, 1.0), (0.4, 2.0), (0.15, 3.0), (0.06, 4.0)]
                for amp, h in harmonics:
                    wave_data += amp * np.sin(2 * np.pi * freq * h * t)

                # ADSR envelope
                attack = min(int(0.01 * self.sample_rate), len(t) // 4)
                release = min(int(0.05 * self.sample_rate), len(t) // 4)
                envelope = np.ones(len(t))
                if attack > 0:
                    envelope[:attack] = np.linspace(0, 1, attack)
                if release > 0:
                    envelope[-release:] = np.linspace(1, 0, release)

                wave_data *= envelope * vel * 0.25

            audio[s:e] += wave_data

        # Normalize per-instrument
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * 0.7

        return audio.astype(np.float32)

    def _classify_instrument(self, instrument: "pretty_midi.Instrument") -> str:
        """Classify an instrument into a guide channel role."""
        name = instrument.name.lower() if instrument.name else ""

        if instrument.is_drum or "drum" in name:
            return "drums"
        elif "bass" in name:
            return "bass"
        elif "lead" in name or "melody" in name:
            return "lead"
        elif "comp" in name or "chord" in name or "piano" in name:
            return "harmony"
        elif "pad" in name or "aux" in name or "texture" in name:
            return "texture"
        else:
            # Classify by program number
            prog = instrument.program
            if prog < 8:
                return "lead"      # Piano family
            elif 24 <= prog < 32:
                return "lead"      # Guitar family
            elif 32 <= prog < 40:
                return "bass"      # Bass family
            elif 88 <= prog < 96:
                return "texture"   # Pad family
            else:
                return "harmony"   # Default

    def _name_to_role(self, name: str) -> TrackRole:
        mapping = {
            "drums": TrackRole.DRUMS,
            "bass": TrackRole.BASS,
            "harmony": TrackRole.COMPING,
            "lead": TrackRole.LEAD,
            "texture": TrackRole.AUX,
        }
        return mapping.get(name, TrackRole.COMPING)

    def _segment_phrases(
        self,
        sequence: Optional[BeatLevelSequence],
        duration: float,
    ) -> List[Tuple[float, float]]:
        """Detect phrase boundaries from the structural trajectory."""
        if sequence is None or len(sequence) == 0:
            # Default: 8-bar phrases
            phrase_dur = 8 * 4 * 60.0 / 120.0  # 8 bars at 4/4, 120 BPM
            phrases = []
            t = 0.0
            while t < duration:
                end = min(t + phrase_dur, duration)
                phrases.append((t, end))
                t = end
            return phrases

        seconds_per_beat = 60.0 / sequence.bpm
        phrases = []
        phrase_start = 0.0

        for i, state in enumerate(sequence):
            if state.boundary_level >= 2 and i > 0:
                phrase_end = i * seconds_per_beat
                if phrase_end > phrase_start:
                    phrases.append((phrase_start, phrase_end))
                phrase_start = phrase_end

        # Last phrase
        final_end = len(sequence) * seconds_per_beat
        if final_end > phrase_start:
            phrases.append((phrase_start, final_end))

        return phrases if phrases else [(0.0, duration)]

    def _save_bundle(self, bundle: GuideBundle, output_dir: str) -> None:
        """Save all guide channels to WAV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save full channels
        for name, channel in bundle.full_channels.items():
            path = os.path.join(output_dir, f"guide_{name}.wav")
            self._save_wav(channel.audio, path)

        # Save phrase segments
        for pg in bundle.phrases:
            phrase_dir = os.path.join(output_dir, f"phrase_{pg.phrase_index:03d}")
            os.makedirs(phrase_dir, exist_ok=True)
            for name, channel in pg.channels.items():
                path = os.path.join(phrase_dir, f"{name}.wav")
                self._save_wav(channel.audio, path)

        # Save metadata
        import json
        meta = {
            "bpm": bundle.bpm,
            "genre": bundle.genre,
            "duration_sec": bundle.duration_sec,
            "n_phrases": len(bundle.phrases),
            "channels": list(bundle.full_channels.keys()),
            "phrases": [
                {
                    "index": pg.phrase_index,
                    "start_sec": pg.start_sec,
                    "end_sec": pg.end_sec,
                    "duration_sec": pg.duration_sec,
                }
                for pg in bundle.phrases
            ],
        }
        with open(os.path.join(output_dir, "guide_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _save_wav(self, audio: np.ndarray, filepath: str) -> None:
        """Save audio array to WAV file."""
        audio_16bit = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_16bit.tobytes())
