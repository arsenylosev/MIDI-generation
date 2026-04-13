"""Revised generation pipeline (v2) integrating the beat-level schema,
sparse candidate generator, learned scorer, multitrack realizer,
guide-audio rendering, and structural evaluation.

This pipeline supersedes the original pipeline.py for new generation runs.
The legacy pipeline is preserved for backward compatibility with the
original piano-roll / SB / diffusion workflow.

Architecture (revised strategy note):
    1. GTTM Prior → beat-level structural plan
    2. Sparse Candidate Generator → musically legal next states
    3. (Optional) Learned Scorer → rank candidates by style plausibility
    4. Schrödinger Bridge → optimal trajectory through waypoints
    5. Multitrack Realizer → concrete MIDI events per track
    6. (Optional) Guide-Audio Renderer → per-stem conditioning signals
    7. (Optional) Structural Evaluator → quality metrics
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from midi_gen.core.config import GenerationConfig
from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence, encode_chord
from midi_gen.candidates.generator import SparseCandidateGenerator
from midi_gen.gttm.structural_prior import (
    GTTMPrior,
    GroupingAnalyzer,
    TensionCurveGenerator,
)
from midi_gen.bridge.schrodinger_bridge import (
    SchrodingerBridgeSolver,
    MusicalState,
)
from midi_gen.realizer.realizer import MultitrackRealizer
from midi_gen.schema.converters import sequence_to_piano_roll
from midi_gen.utils.midi_utils import (
    generate_form_string,
    create_whole_song_gen_input,
)


class MidiGenerationPipelineV2:
    """Revised end-to-end pipeline for MIDI generation.

    Combines the beat-level structural representation with the original
    GTTM + Schrödinger Bridge approach, adding sparse candidate generation,
    optional learned scoring, multitrack realization, guide-audio rendering,
    and structural evaluation.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig.prog_rock()
        self.rng = np.random.default_rng(self.config.seed)

        # Core components
        self.gttm_prior = GTTMPrior(
            config=self.config.gttm,
            edo=self.config.tonal.edo,
            genre=self.config.genre,
        )
        self.grouping = GroupingAnalyzer(genre=self.config.genre)
        self.tension_gen = TensionCurveGenerator(
            resolution=self.config.gttm.tension_curve_resolution,
        )
        self.candidate_gen = SparseCandidateGenerator(
            genre=self.config.genre,
            max_candidates=self.config.candidates.max_candidates,
        )
        self.sb_solver = SchrodingerBridgeSolver(
            energy_fn=self.gttm_prior.compute_transition_energy,
            num_steps=self.config.bridge.num_diffusion_steps,
            sinkhorn_iterations=self.config.bridge.sinkhorn_iterations,
            sinkhorn_epsilon=self.config.bridge.sinkhorn_epsilon,
            num_candidates=self.config.bridge.num_candidates_per_step,
            genre=self.config.genre,
        )
        self.realizer = MultitrackRealizer(
            genre=self.config.genre,
            velocity_base=self.config.realizer.velocity_base,
        )

        # Optional scorer
        self.scorer = None
        if self.config.scorer.use_scorer:
            try:
                from midi_gen.scorer.model import CandidateScorer
                self.scorer = CandidateScorer(
                    embed_dim=self.config.scorer.embed_dim,
                    num_heads=self.config.scorer.num_heads,
                    num_layers=self.config.scorer.num_layers,
                    context_length=self.config.scorer.context_length,
                )
                if self.config.scorer.checkpoint_path:
                    self.scorer.load(self.config.scorer.checkpoint_path)
            except ImportError:
                pass

        if self.config.verbose:
            print(f"[MidiGenV2] Pipeline initialized for genre: {self.config.genre}")
            print(f"[MidiGenV2] Scorer: {'enabled' if self.scorer else 'disabled'}")
            print(f"[MidiGenV2] Realizer: {'enabled' if self.config.realizer.enabled else 'disabled'}")

    def generate(
        self,
        num_measures: Optional[int] = None,
        key: Optional[int] = None,
        is_major: Optional[bool] = None,
        phrase_string: Optional[str] = None,
        bpm: Optional[float] = None,
        output_dir: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[Dict]:
        """Generate one or more MIDI pieces using the revised pipeline."""
        if num_measures is None:
            num_measures = int(self.rng.integers(
                self.config.structure.min_measures,
                self.config.structure.max_measures + 1,
            ))
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
                print(f"\n[MidiGenV2] Generating sample {i + 1}/{num_samples}...")
            result = self._generate_single(
                num_measures, key, is_major, phrase_string, bpm, output_dir, i,
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
        sample_idx: int,
    ) -> Dict:
        """Generate a single piece through the revised pipeline."""
        beats_per_measure = self.config.structure.num_beats_per_measure
        steps_per_beat = self.config.structure.num_steps_per_beat

        # --- Step 1: Phrase structure ---
        if self.config.verbose:
            print("[MidiGenV2] Step 1: Generating phrase structure...")

        if phrase_string:
            phrases = self._parse_phrase_string(phrase_string)
            num_measures = sum(p["lgth"] for p in phrases)
        else:
            phrases = self.grouping.generate_phrase_structure(num_measures, self.rng)

        form_string = generate_form_string(phrases)

        # --- Step 2: Beat-level structural trajectory ---
        if self.config.verbose:
            print("[MidiGenV2] Step 2: Building beat-level trajectory...")

        sequence = self._build_beat_sequence(
            phrases, key, is_major, bpm, beats_per_measure,
        )

        # --- Step 3: Tension curve ---
        if self.config.verbose:
            print("[MidiGenV2] Step 3: Generating tension curve...")

        tension_curve = self.tension_gen.generate_curve(
            phrases, beats_per_measure, steps_per_beat,
        )

        # --- Step 4: SB trajectory (on waypoints) ---
        if self.config.verbose:
            print("[MidiGenV2] Step 4: Computing Schrödinger Bridge trajectory...")

        waypoints = self._create_waypoints(
            phrases, key, is_major, beats_per_measure, steps_per_beat,
        )
        trajectory = self.sb_solver.solve_with_waypoints(
            waypoints, tension_curve, self.rng,
        )

        # --- Step 5: Apply trajectory guidance to beat-level sequence ---
        if self.config.verbose:
            print("[MidiGenV2] Step 5: Applying trajectory guidance...")

        sequence = self._apply_trajectory_to_sequence(sequence, trajectory)

        # --- Step 6: Multitrack realization ---
        if self.config.verbose:
            print("[MidiGenV2] Step 6: Multitrack realization...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = os.path.join(
            output_dir, f"{self.config.genre}_{timestamp}_s{sample_idx}",
        )
        os.makedirs(sample_dir, exist_ok=True)

        midi_path = os.path.join(sample_dir, "generation.mid")
        if self.config.realizer.enabled:
            pm = self.realizer.realize(
                sequence, tension_curve=self._resample_tension(tension_curve, len(sequence)),
                output_path=midi_path,
            )
        else:
            # Fallback: convert sequence to piano-roll and use legacy path
            piano_roll = sequence_to_piano_roll(sequence)
            from midi_gen.utils.midi_utils import (
                piano_roll_to_note_list, note_list_to_midi, save_midi,
            )
            notes = piano_roll_to_note_list(piano_roll, bpm=bpm, steps_per_beat=steps_per_beat)
            midi_obj = note_list_to_midi(notes, bpm=bpm)
            save_midi(midi_obj, midi_path)
            pm = midi_obj

        # --- Step 7: Save metadata ---
        metadata = {
            "genre": self.config.genre,
            "form_string": form_string,
            "key": key,
            "is_major": is_major,
            "bpm": bpm,
            "num_measures": num_measures,
            "num_beats": len(sequence),
            "phrases": phrases,
            "pipeline_version": "v2",
            "timestamp": timestamp,
            "config": {
                "scorer_enabled": self.scorer is not None,
                "realizer_enabled": self.config.realizer.enabled,
                "guide_rendering_enabled": self.config.guide_rendering.enabled,
            },
        }
        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save beat-level sequence
        sequence.to_json(os.path.join(sample_dir, "beat_sequence.json"))

        # Save tension curve
        np.save(os.path.join(sample_dir, "tension_curve.npy"), tension_curve)

        result = {
            "midi_path": midi_path,
            "metadata_path": metadata_path,
            "sample_dir": sample_dir,
            "form_string": form_string,
            "num_beats": len(sequence),
            "duration_seconds": pm.get_end_time() if pm else 0,
            "sequence": sequence,
        }

        # --- Step 8: Optional guide-audio rendering ---
        if self.config.guide_rendering.enabled:
            if self.config.verbose:
                print("[MidiGenV2] Step 8: Rendering guide audio...")
            try:
                from midi_gen.rendering.guide_renderer import GuideAudioRenderer
                guide_dir = os.path.join(sample_dir, "guides")
                guide_renderer = GuideAudioRenderer(
                    sample_rate=self.config.guide_rendering.sample_rate,
                )
                bundle = guide_renderer.render_guides(
                    midi_path, sequence=sequence, output_dir=guide_dir,
                )
                result["guide_dir"] = guide_dir
                result["guide_channels"] = list(bundle.full_channels.keys())
            except Exception as e:
                if self.config.verbose:
                    print(f"[MidiGenV2] Guide rendering failed: {e}")

        # --- Step 9: Optional structural evaluation ---
        if self.config.evaluation.enabled:
            if self.config.verbose:
                print("[MidiGenV2] Step 9: Running structural evaluation...")
            try:
                from midi_gen.evaluation.structural_metrics import StructuralEvaluator
                evaluator = StructuralEvaluator()
                report = evaluator.evaluate(sequence)
                result["evaluation"] = report.scores
                result["evaluation_overall"] = report.overall_score

                eval_path = os.path.join(sample_dir, "evaluation.json")
                with open(eval_path, "w") as f:
                    json.dump({
                        "scores": report.scores,
                        "overall": report.overall_score,
                    }, f, indent=2)

                if self.config.verbose:
                    print(report.summary())
            except Exception as e:
                if self.config.verbose:
                    print(f"[MidiGenV2] Evaluation failed: {e}")

        # --- Step 10: WSG compatibility output ---
        piano_roll = sequence_to_piano_roll(sequence)
        wsg_input = create_whole_song_gen_input(piano_roll, phrases, key, is_major)
        result["whole_song_gen_input"] = wsg_input

        if self.config.verbose:
            print(f"[MidiGenV2] Generation complete! Output: {midi_path}")

        return result

    # --- Helper methods ---

    def _build_beat_sequence(
        self,
        phrases: List[Dict],
        key: int,
        is_major: bool,
        bpm: float,
        beats_per_measure: int,
    ) -> BeatLevelSequence:
        """Build a beat-level structural trajectory from phrase structure."""
        states = []

        section_chords = {
            "intro": (key, 0 if is_major else 1),
            "theme_a": (key, 0 if is_major else 1),
            "verse": (key, 0 if is_major else 1),
            "theme_b": ((key + 7) % 12, 0),
            "chorus": ((key + 5) % 12, 0),
            "development": ((key + 5) % 12, 0),
            "bridge": ((key + 9) % 12, 1),
            "solo": ((key + 2) % 12, 4),
            "coda": (key, 0 if is_major else 1),
            "outro": (key, 0 if is_major else 1),
            "head_in": (key, 3),
            "head_out": (key, 3),
            "solo_section": ((key + 2) % 12, 4),
            "interlude": ((key + 9) % 12, 1),
            "trading_fours": ((key + 7) % 12, 2),
        }

        region_idx = 0
        for phrase in phrases:
            root, quality = section_chords.get(phrase["name"], (key, 0))
            chord_label = encode_chord(root, quality)
            n_bars = phrase["lgth"]

            for bar in range(n_bars):
                for beat in range(beats_per_measure):
                    boundary = 0
                    if bar == 0 and beat == 0:
                        boundary = 3  # section boundary
                    elif beat == 0 and bar % 4 == 0:
                        boundary = 2  # phrase boundary

                    # Harmonic role: tonic at start/end, dominant before boundaries
                    role = 0
                    if bar == n_bars - 1 and beat == beats_per_measure - 1:
                        role = 2  # dominant before next section
                    elif bar == n_bars - 2 and beat == 0:
                        role = 1  # pre-dominant

                    states.append(BeatLevelState(
                        meter_token=0,  # 4/4 default
                        beat_position=beat,
                        boundary_level=boundary,
                        region_label=region_idx,
                        chord_label=chord_label,
                        harmonic_role=role,
                        melodic_head=60 + (beat % 4),  # placeholder
                        groove_token=0,
                    ))

            region_idx += 1

        return BeatLevelSequence(
            states=states,
            bpm=bpm,
            key=key,
            is_major=is_major,
            genre=self.config.genre,
            form_string=generate_form_string(phrases),
        )

    def _apply_trajectory_to_sequence(
        self,
        sequence: BeatLevelSequence,
        trajectory: List[MusicalState],
    ) -> BeatLevelSequence:
        """Apply SB trajectory guidance to the beat-level sequence."""
        if not trajectory or len(trajectory) < 2:
            return sequence

        n = len(sequence)
        traj_len = len(trajectory)
        new_states = []

        for i, state in enumerate(sequence):
            traj_idx = min(int(i * traj_len / n), traj_len - 1)
            traj_state = trajectory[traj_idx]

            # Update chord from trajectory
            new_chord = encode_chord(traj_state.chord_root, traj_state.chord_quality)

            # Update melodic head from trajectory pitch vector
            mel = state.melodic_head
            if hasattr(traj_state, 'pitch_vector') and traj_state.pitch_vector is not None:
                active = np.where(traj_state.pitch_vector > 0.5)[0]
                if len(active) > 0:
                    # Pick the highest active pitch in melody range (60-84)
                    melody_range = [p for p in active if 60 <= p <= 84]
                    if melody_range:
                        mel = melody_range[-1]

            new_states.append(BeatLevelState(
                meter_token=state.meter_token,
                beat_position=state.beat_position,
                boundary_level=state.boundary_level,
                region_label=state.region_label,
                chord_label=new_chord,
                harmonic_role=state.harmonic_role,
                melodic_head=mel,
                groove_token=state.groove_token,
            ))

        return BeatLevelSequence(
            states=new_states,
            bpm=sequence.bpm,
            key=sequence.key,
            is_major=sequence.is_major,
            genre=sequence.genre,
            form_string=sequence.form_string,
        )

    def _create_waypoints(
        self,
        phrases: List[Dict],
        key: int,
        is_major: bool,
        beats_per_measure: int,
        steps_per_beat: int,
    ) -> List[MusicalState]:
        """Create structural waypoints at phrase boundaries (same as legacy)."""
        waypoints = []
        current_step = 0

        section_chords = {
            "intro": (key, 3 if not is_major else 0),
            "theme_a": (key, 3 if not is_major else 0),
            "theme_b": ((key + 7) % 12, 0),
            "development": ((key + 5) % 12, 0),
            "solo": ((key + 2) % 12, 4),
            "coda": (key, 3 if not is_major else 0),
            "head_in": (key, 3 if not is_major else 0),
            "head_out": (key, 3 if not is_major else 0),
            "interlude": ((key + 9) % 12, 1),
            "trading_fours": ((key + 7) % 12, 2),
        }

        for phrase in phrases:
            chord_root, chord_qual = section_chords.get(phrase["name"], (key, 0))
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
                metrical_weight=1.0,
            ))
            phrase_steps = phrase["lgth"] * beats_per_measure * steps_per_beat
            current_step += phrase_steps

        # Final tonic waypoint
        final_vec = np.zeros(128)
        for octave in [3, 4, 5]:
            for interval in [0, 4 if is_major else 3, 7]:
                p = octave * 12 + key + interval
                if 0 <= p < 128:
                    final_vec[p] = 1.0

        waypoints.append(MusicalState(
            time_step=current_step,
            pitch_vector=final_vec,
            chord_root=key,
            chord_quality=0 if is_major else 1,
            tension=0.1,
            velocity=0.5,
            metrical_weight=1.0,
        ))
        return waypoints

    def _resample_tension(
        self, tension_curve: np.ndarray, target_len: int,
    ) -> np.ndarray:
        """Resample a tension curve to match the beat-level sequence length."""
        if len(tension_curve) == target_len:
            return tension_curve
        indices = np.linspace(0, len(tension_curve) - 1, target_len)
        return np.interp(indices, np.arange(len(tension_curve)), tension_curve)

    def _parse_phrase_string(self, phrase_string: str) -> List[Dict]:
        """Parse a whole-song-gen format phrase string."""
        phrases = []
        type_names = {
            "i": "intro", "A": "theme_a", "B": "theme_b",
            "C": "development", "S": "solo", "b": "bridge",
            "o": "coda", "T": "trading_fours",
        }
        i = 0
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
                "name": name, "type": ptype,
                "start": sum(p["lgth"] for p in phrases),
                "lgth": length,
            })
        return phrases

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "MidiGenerationPipelineV2":
        """Create a pipeline from a named preset."""
        if preset == "prog_rock":
            config = GenerationConfig.prog_rock()
        elif preset == "jazz_fusion":
            config = GenerationConfig.jazz_fusion()
        else:
            raise ValueError(f"Unknown preset: {preset}")
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return cls(config)

    @classmethod
    def from_config_file(cls, path: str) -> "MidiGenerationPipelineV2":
        """Create a pipeline from a JSON configuration file."""
        config = GenerationConfig.from_json(path)
        return cls(config)
