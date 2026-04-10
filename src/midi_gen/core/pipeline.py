"""
Main Generation Pipeline for MIDI Music Generation.

Orchestrates the complete generation process:
1. Structure planning (GTTM-informed phrase structure)
2. Tension curve generation (prolongational reduction)
3. Schrödinger Bridge trajectory solving
4. Diffusion-based piano-roll generation
5. MIDI output and optional audio rendering

This pipeline produces output compatible with the whole-song-gen
cascaded diffusion architecture.
"""

import numpy as np
import os
import json
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from midi_gen.core.config import GenerationConfig
from midi_gen.gttm.structural_prior import (
    GTTMPrior, GroupingAnalyzer, TensionCurveGenerator,
    MetricalGrid, TonalDistanceMetric, BeatState
)
from midi_gen.bridge.schrodinger_bridge import (
    SchrodingerBridgeSolver, MusicalState, CandidateGenerator
)
from midi_gen.models.diffusion_model import MidiDiffusionModel, DiffusionConfig
from midi_gen.utils.midi_utils import (
    trajectory_to_piano_roll, piano_roll_to_note_list,
    note_list_to_midi, save_midi, generate_form_string,
    create_whole_song_gen_input
)


class MidiGenerationPipeline:
    """
    End-to-end pipeline for generating MIDI music in progressive rock
    and jazz fusion styles.

    The pipeline combines GTTM structural analysis, Schrödinger Bridge
    inference, and diffusion-based generation to produce structurally
    coherent, stylistically authentic music.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig.prog_rock()
        self.rng = np.random.default_rng(self.config.seed)

        # Initialize components
        self.gttm_prior = GTTMPrior(
            config=self.config.gttm,
            edo=self.config.tonal.edo,
            genre=self.config.genre
        )
        self.grouping = GroupingAnalyzer(genre=self.config.genre)
        self.tension_gen = TensionCurveGenerator(
            resolution=self.config.gttm.tension_curve_resolution
        )
        self.diffusion = MidiDiffusionModel(
            DiffusionConfig(
                num_steps=self.config.bridge.num_diffusion_steps,
                latent_dim=self.config.model.latent_dim,
                hidden_dim=self.config.model.hidden_dim,
                num_layers=self.config.model.num_layers
            )
        )

        # Initialize SB solver with GTTM energy function
        self.sb_solver = SchrodingerBridgeSolver(
            energy_fn=self.gttm_prior.compute_transition_energy,
            num_steps=self.config.bridge.num_diffusion_steps,
            sinkhorn_iterations=self.config.bridge.sinkhorn_iterations,
            sinkhorn_epsilon=self.config.bridge.sinkhorn_epsilon,
            num_candidates=self.config.bridge.num_candidates_per_step,
            genre=self.config.genre
        )

        if self.config.verbose:
            print(f"[MidiGen] Pipeline initialized for genre: {self.config.genre}")
            print(f"[MidiGen] Tonal system: {self.config.tonal.edo}-EDO")
            print(f"[MidiGen] Device: {self.config.device}")

    def generate(
        self,
        num_measures: Optional[int] = None,
        key: Optional[int] = None,
        is_major: Optional[bool] = None,
        phrase_string: Optional[str] = None,
        bpm: Optional[float] = None,
        output_dir: Optional[str] = None,
        num_samples: int = 1
    ) -> List[Dict]:
        """
        Generate one or more MIDI pieces.

        Args:
            num_measures: Total number of measures (default: random 32-64)
            key: Tonic pitch class 0-11 (default: from config)
            is_major: Major or minor key (default: from config)
            phrase_string: Optional whole-song-gen format phrase string
            bpm: Tempo in BPM (default: from config)
            output_dir: Directory for output files
            num_samples: Number of pieces to generate

        Returns:
            List of generation result dictionaries
        """
        if num_measures is None:
            num_measures = self.rng.integers(
                self.config.structure.min_measures,
                self.config.structure.max_measures + 1
            )
        if key is None:
            key = self.config.tonal.default_key
        if is_major is None:
            is_major = self.config.tonal.is_major
        if bpm is None:
            bpm = self.config.structure.bpm
        if output_dir is None:
            output_dir = self.config.output_dir

        results = []
        for i in range(num_samples):
            if self.config.verbose:
                print(f"\n[MidiGen] Generating sample {i + 1}/{num_samples}...")

            result = self._generate_single(
                num_measures, key, is_major, phrase_string, bpm, output_dir, i
            )
            results.append(result)

        return results

    def _generate_single(
        self,
        num_measures: int,
        key: int,
        is_major: bool,
        phrase_string: Optional[str],
        bpm: float,
        output_dir: str,
        sample_idx: int
    ) -> Dict:
        """Generate a single piece."""
        beats_per_measure = self.config.structure.num_beats_per_measure
        steps_per_beat = self.config.structure.num_steps_per_beat

        # Step 1: Generate phrase structure
        if self.config.verbose:
            print("[MidiGen] Step 1: Generating phrase structure...")

        if phrase_string:
            phrases = self._parse_phrase_string(phrase_string)
            num_measures = sum(p["lgth"] for p in phrases)
        else:
            phrases = self.grouping.generate_phrase_structure(
                num_measures, self.rng
            )

        form_string = generate_form_string(phrases)
        if self.config.verbose:
            print(f"[MidiGen]   Form: {form_string}")
            print(f"[MidiGen]   Key: {key} ({'major' if is_major else 'minor'})")
            print(f"[MidiGen]   Measures: {num_measures}")

        # Step 2: Generate tension curve
        if self.config.verbose:
            print("[MidiGen] Step 2: Generating tension curve...")

        tension_curve = self.tension_gen.generate_curve(
            phrases, beats_per_measure, steps_per_beat
        )

        # Step 3: Create structural waypoints via SB
        if self.config.verbose:
            print("[MidiGen] Step 3: Computing Schrödinger Bridge trajectory...")

        waypoints = self._create_waypoints(
            phrases, key, is_major, beats_per_measure, steps_per_beat
        )

        trajectory = self.sb_solver.solve_with_waypoints(
            waypoints, tension_curve, self.rng
        )

        # Step 4: Generate detailed piano-roll via diffusion
        if self.config.verbose:
            print("[MidiGen] Step 4: Generating piano-roll via diffusion...")

        piano_roll = self.diffusion.generate_conditioned(
            phrase_structure=phrases,
            key=key,
            is_major=is_major,
            beats_per_measure=beats_per_measure,
            steps_per_beat=steps_per_beat,
            tension_curve=tension_curve,
            rng=self.rng
        )

        # Step 5: Apply trajectory guidance to piano-roll
        if self.config.verbose:
            print("[MidiGen] Step 5: Applying structural guidance...")

        guided_roll = self._apply_trajectory_guidance(
            piano_roll, trajectory, key, is_major
        )

        # Step 6: Convert to MIDI and save
        if self.config.verbose:
            print("[MidiGen] Step 6: Saving MIDI output...")

        result = self._save_output(
            guided_roll, phrases, key, is_major, bpm,
            tension_curve, output_dir, sample_idx
        )

        # Step 7: Create whole-song-gen compatible output
        wsg_input = create_whole_song_gen_input(
            guided_roll, phrases, key, is_major
        )
        result["whole_song_gen_input"] = wsg_input

        if self.config.verbose:
            print(f"[MidiGen] Generation complete! Output: {result['midi_path']}")

        return result

    def _create_waypoints(
        self,
        phrases: List[Dict],
        key: int,
        is_major: bool,
        beats_per_measure: int,
        steps_per_beat: int
    ) -> List[MusicalState]:
        """Create structural waypoints at phrase boundaries."""
        waypoints = []
        current_step = 0

        # Chord progressions typical for each section type
        section_chords = {
            "intro": (key, 3 if not is_major else 0),       # tonic
            "theme_a": (key, 3 if not is_major else 0),     # tonic
            "theme_b": ((key + 7) % 12, 0),                 # dominant
            "development": ((key + 5) % 12, 0),             # subdominant
            "solo": ((key + 2) % 12, 4),                    # supertonic min7
            "coda": (key, 3 if not is_major else 0),        # tonic
            "head_in": (key, 3 if not is_major else 0),
            "head_out": (key, 3 if not is_major else 0),
            "interlude": ((key + 9) % 12, 1),
            "trading_fours": ((key + 7) % 12, 2),
        }

        for phrase in phrases:
            chord_root, chord_qual = section_chords.get(
                phrase["name"], (key, 0)
            )

            # Create pitch vector for the waypoint chord
            pitch_vec = np.zeros(128)
            intervals = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 4, 7, 10],
                         3: [0, 4, 7, 11], 4: [0, 3, 7, 10]}
            for octave in [3, 4, 5]:
                for interval in intervals.get(chord_qual, [0, 4, 7]):
                    p = octave * 12 + chord_root + interval
                    if 0 <= p < 128:
                        pitch_vec[p] = 1.0

            waypoints.append(MusicalState(
                time_step=current_step,
                pitch_vector=pitch_vec,
                chord_root=chord_root,
                chord_quality=chord_qual,
                tension=0.3,
                velocity=0.7,
                metrical_weight=1.0
            ))

            phrase_steps = phrase["lgth"] * beats_per_measure * steps_per_beat
            current_step += phrase_steps

        # Add final waypoint (return to tonic)
        final_pitch_vec = np.zeros(128)
        for octave in [3, 4, 5]:
            for interval in [0, 4, 7] if is_major else [0, 3, 7]:
                p = octave * 12 + key + interval
                if 0 <= p < 128:
                    final_pitch_vec[p] = 1.0

        waypoints.append(MusicalState(
            time_step=current_step,
            pitch_vector=final_pitch_vec,
            chord_root=key,
            chord_quality=0 if is_major else 1,
            tension=0.1,
            velocity=0.5,
            metrical_weight=1.0
        ))

        return waypoints

    def _apply_trajectory_guidance(
        self,
        piano_roll: np.ndarray,
        trajectory: List[MusicalState],
        key: int,
        is_major: bool
    ) -> np.ndarray:
        """
        Apply the SB trajectory as structural guidance to the
        diffusion-generated piano-roll.

        This blends the detailed texture from diffusion with the
        harmonic structure from the SB trajectory.
        """
        guided = piano_roll.copy()
        num_steps = piano_roll.shape[1]

        # Resample trajectory to match piano-roll length
        traj_len = len(trajectory)
        if traj_len < 2:
            return guided

        for t in range(num_steps):
            # Find corresponding trajectory state
            traj_idx = min(int(t * traj_len / num_steps), traj_len - 1)
            state = trajectory[traj_idx]

            # Boost pitches that align with the trajectory chord
            for p in range(128):
                if state.pitch_vector[p] > 0.5:
                    # Boost onset probability for chord tones
                    guided[0, t, p] = max(guided[0, t, p], 0.3 * state.tension)

            # Apply key-based filtering: reduce non-diatonic notes
            scale = self._get_scale(key, is_major)
            for p in range(128):
                if (p % 12) not in scale:
                    guided[0, t, p] *= 0.3
                    guided[1, t, p] *= 0.3

        # Re-threshold
        guided[0] = (guided[0] > 0.4).astype(float)
        guided[1] = (guided[1] > 0.3).astype(float)

        return guided

    def _get_scale(self, key: int, is_major: bool) -> set:
        """Get the pitch classes of a major or minor scale."""
        if is_major:
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:
            intervals = [0, 2, 3, 5, 7, 8, 10]
        return {(key + i) % 12 for i in intervals}

    def _parse_phrase_string(self, phrase_string: str) -> List[Dict]:
        """Parse a whole-song-gen format phrase string."""
        phrases = []
        i = 0
        type_names = {
            "i": "intro", "A": "theme_a", "B": "theme_b",
            "C": "development", "S": "solo", "b": "bridge",
            "o": "coda", "T": "trading_fours"
        }

        while i < len(phrase_string):
            ptype = phrase_string[i]
            i += 1
            num_str = ""
            while i < len(phrase_string) and phrase_string[i].isdigit():
                num_str += phrase_string[i]
                i += 1

            length = int(num_str) if num_str else 4
            name = type_names.get(ptype, "theme_a")

            phrases.append({
                "name": name,
                "type": ptype,
                "start": sum(p["lgth"] for p in phrases),
                "lgth": length
            })

        return phrases

    def _save_output(
        self,
        piano_roll: np.ndarray,
        phrases: List[Dict],
        key: int,
        is_major: bool,
        bpm: float,
        tension_curve: np.ndarray,
        output_dir: str,
        sample_idx: int
    ) -> Dict:
        """Save generated output to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = os.path.join(
            output_dir,
            f"{self.config.genre}_{timestamp}_s{sample_idx}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        # Save MIDI
        notes = piano_roll_to_note_list(
            piano_roll, bpm=bpm,
            steps_per_beat=self.config.structure.num_steps_per_beat
        )
        midi_obj = note_list_to_midi(notes, bpm=bpm)
        midi_path = os.path.join(sample_dir, "generation.mid")
        save_midi(midi_obj, midi_path)

        # Save metadata
        form_string = generate_form_string(phrases)
        metadata = {
            "genre": self.config.genre,
            "form_string": form_string,
            "key": key,
            "is_major": is_major,
            "bpm": bpm,
            "num_measures": sum(p["lgth"] for p in phrases),
            "phrases": phrases,
            "timestamp": timestamp,
            "config": {
                "edo": self.config.tonal.edo,
                "bridge_steps": self.config.bridge.num_diffusion_steps,
                "sinkhorn_iterations": self.config.bridge.sinkhorn_iterations,
            }
        }
        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save form description
        form_path = os.path.join(sample_dir, "form.txt")
        with open(form_path, "w") as f:
            f.write(f"Form: {form_string}\n")
            f.write(f"Key: {key} ({'major' if is_major else 'minor'})\n")
            f.write(f"BPM: {bpm}\n")
            for phrase in phrases:
                f.write(f"  {phrase['name']}: {phrase['lgth']} measures\n")

        # Save piano-roll as numpy
        np.save(os.path.join(sample_dir, "piano_roll.npy"), piano_roll)

        # Save tension curve
        np.save(os.path.join(sample_dir, "tension_curve.npy"), tension_curve)

        return {
            "midi_path": midi_path,
            "metadata_path": metadata_path,
            "form_path": form_path,
            "piano_roll_path": os.path.join(sample_dir, "piano_roll.npy"),
            "sample_dir": sample_dir,
            "form_string": form_string,
            "num_notes": len(notes),
            "duration_seconds": midi_obj.get_end_time() if notes else 0,
        }

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "MidiGenerationPipeline":
        """
        Create a pipeline from a named preset.

        Available presets: 'prog_rock', 'jazz_fusion'
        """
        if preset == "prog_rock":
            config = GenerationConfig.prog_rock()
        elif preset == "jazz_fusion":
            config = GenerationConfig.jazz_fusion()
        else:
            raise ValueError(f"Unknown preset: {preset}. Use 'prog_rock' or 'jazz_fusion'.")

        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        return cls(config)

    @classmethod
    def from_config_file(cls, path: str) -> "MidiGenerationPipeline":
        """Create a pipeline from a JSON configuration file."""
        config = GenerationConfig.from_json(path)
        return cls(config)
