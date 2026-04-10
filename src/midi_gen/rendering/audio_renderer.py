"""
MIDI-to-Audio Rendering Module.

Provides multiple backends for converting generated MIDI files to audio:
1. FluidSynth + SoundFont (highest quality, requires system FluidSynth)
2. Pure Python synthesis (fallback, basic sine-wave rendering)

The FluidSynth backend supports high-quality SoundFont (.sf2) files
for realistic instrument rendering, while the pure Python fallback
ensures the module works in any environment.
"""

import numpy as np
import os
import struct
import wave
from typing import Optional, List, Dict, Tuple


class AudioRenderer:
    """
    Multi-backend audio renderer for MIDI files.

    Automatically selects the best available backend:
    1. FluidSynth (via midi2audio or pyfluidsynth)
    2. Pure Python sine-wave synthesis (fallback)
    """

    def __init__(
        self,
        soundfont_path: Optional[str] = None,
        sample_rate: int = 44100,
        output_format: str = "wav"
    ):
        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate
        self.output_format = output_format
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect the best available audio rendering backend."""
        # Try FluidSynth via midi2audio
        try:
            from midi2audio import FluidSynth
            if self.soundfont_path and os.path.exists(self.soundfont_path):
                return "fluidsynth"
            # Try to find a default soundfont
            default_paths = [
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/usr/share/soundfonts/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/default-GM.sf2",
                os.path.expanduser("~/.local/share/soundfonts/default.sf2"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self.soundfont_path = path
                    return "fluidsynth"
        except ImportError:
            pass

        # Try pyfluidsynth directly
        try:
            import fluidsynth
            return "pyfluidsynth"
        except ImportError:
            pass

        # Fallback to pure Python
        return "python"

    def render(
        self,
        midi_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Render a MIDI file to audio.

        Args:
            midi_path: Path to the input MIDI file
            output_path: Path for the output audio file (auto-generated if None)

        Returns:
            Path to the rendered audio file
        """
        if output_path is None:
            base = os.path.splitext(midi_path)[0]
            output_path = f"{base}.{self.output_format}"

        if self.backend == "fluidsynth":
            return self._render_fluidsynth(midi_path, output_path)
        elif self.backend == "pyfluidsynth":
            return self._render_pyfluidsynth(midi_path, output_path)
        else:
            return self._render_python(midi_path, output_path)

    def _render_fluidsynth(self, midi_path: str, output_path: str) -> str:
        """Render using FluidSynth via midi2audio."""
        from midi2audio import FluidSynth
        fs = FluidSynth(self.soundfont_path, sample_rate=self.sample_rate)
        fs.midi_to_audio(midi_path, output_path)
        return output_path

    def _render_pyfluidsynth(self, midi_path: str, output_path: str) -> str:
        """Render using pyfluidsynth directly."""
        import fluidsynth

        synth = fluidsynth.Synth(samplerate=float(self.sample_rate))
        if self.soundfont_path:
            sfid = synth.sfload(self.soundfont_path)
            synth.program_select(0, sfid, 0, 0)

        # Load and play MIDI
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
            audio = midi.fluidsynth(
                fs=self.sample_rate,
                sf2_path=self.soundfont_path
            )
            self._save_wav(audio, output_path)
        finally:
            synth.delete()

        return output_path

    def _render_python(self, midi_path: str, output_path: str) -> str:
        """
        Render using pure Python sine-wave synthesis.

        This is a basic fallback that produces recognizable but
        not realistic audio output.
        """
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
        except ImportError:
            raise ImportError(
                "pretty_midi is required for Python rendering. "
                "Install with: pip install pretty_midi"
            )

        duration = midi.get_end_time() + 1.0
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples)

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                audio = self._add_note_to_audio(
                    audio, note.pitch, note.start, note.end,
                    note.velocity / 127.0
                )

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        self._save_wav(audio, output_path)
        return output_path

    def _add_note_to_audio(
        self,
        audio: np.ndarray,
        pitch: int,
        start: float,
        end: float,
        velocity: float
    ) -> np.ndarray:
        """Add a single note to the audio buffer using additive synthesis."""
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        start_sample = int(start * self.sample_rate)
        end_sample = min(int(end * self.sample_rate), len(audio))

        if start_sample >= end_sample:
            return audio

        t = np.arange(end_sample - start_sample) / self.sample_rate

        # Additive synthesis: fundamental + harmonics
        wave = np.zeros_like(t)
        harmonics = [(1.0, 1.0), (0.5, 2.0), (0.25, 3.0), (0.125, 4.0)]
        for amp, harmonic in harmonics:
            wave += amp * np.sin(2 * np.pi * freq * harmonic * t)

        # ADSR envelope
        attack = min(0.01, len(t) / self.sample_rate * 0.1)
        release = min(0.05, len(t) / self.sample_rate * 0.2)
        envelope = np.ones_like(t)

        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0 and release_samples < len(envelope):
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)

        wave *= envelope * velocity * 0.3
        audio[start_sample:end_sample] += wave

        return audio

    def _save_wav(self, audio: np.ndarray, filepath: str):
        """Save audio array to WAV file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Convert to 16-bit PCM
        audio_16bit = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_16bit.tobytes())

    def get_backend_info(self) -> Dict:
        """Return information about the current rendering backend."""
        return {
            "backend": self.backend,
            "soundfont": self.soundfont_path,
            "sample_rate": self.sample_rate,
            "output_format": self.output_format,
        }


def render_midi_to_audio(
    midi_path: str,
    output_path: Optional[str] = None,
    soundfont_path: Optional[str] = None,
    sample_rate: int = 44100
) -> str:
    """
    Convenience function to render a MIDI file to audio.

    Automatically selects the best available backend.
    """
    renderer = AudioRenderer(
        soundfont_path=soundfont_path,
        sample_rate=sample_rate
    )
    return renderer.render(midi_path, output_path)
