"""
GTTM Structural Prior for Music Generation.

Implements computable features derived from Lerdahl and Jackendoff's
Generative Theory of Tonal Music (GTTM), including grouping structure,
metrical structure, time-span reduction, and prolongational reduction.

These features are combined into an energy function that serves as the
structural prior for the Schrödinger Bridge inference.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BeatState:
    """Represents the musical state at a single beat position."""
    beat_index: int
    pitch_classes: np.ndarray  # (edo,) binary vector of active pitch classes
    bass_pitch_class: int
    chord_quality: str  # "maj", "min", "dom7", "dim", "aug", "sus4", etc.
    metrical_weight: float
    grouping_level: int  # 0=sub-beat, 1=beat, 2=measure, 3=phrase, 4=section
    tension: float  # 0.0 (relaxed) to 1.0 (tense)
    velocity: float  # 0.0 to 1.0


class TonalDistanceMetric:
    """
    Computes tonal distance between pitch-class sets using the
    Lerdahl (2001) tonal pitch space model.

    The metric captures the cognitive distance between chords and keys,
    which is essential for modeling tension and relaxation in GTTM.
    """

    # Circle of fifths distance matrix for 12-EDO
    FIFTHS_DISTANCE_12 = np.array([
        [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5],  # C
        [5, 0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2],  # C#
        [2, 5, 0, 5, 2, 3, 4, 1, 6, 1, 4, 3],  # D
        [3, 2, 5, 0, 5, 2, 3, 4, 1, 6, 1, 4],  # Eb
        [4, 3, 2, 5, 0, 5, 2, 3, 4, 1, 6, 1],  # E
        [1, 4, 3, 2, 5, 0, 5, 2, 3, 4, 1, 6],  # F
        [6, 1, 4, 3, 2, 5, 0, 5, 2, 3, 4, 1],  # F#
        [1, 6, 1, 4, 3, 2, 5, 0, 5, 2, 3, 4],  # G
        [4, 1, 6, 1, 4, 3, 2, 5, 0, 5, 2, 3],  # Ab
        [3, 4, 1, 6, 1, 4, 3, 2, 5, 0, 5, 2],  # A
        [2, 3, 4, 1, 6, 1, 4, 3, 2, 5, 0, 5],  # Bb
        [5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5, 0],  # B
    ])

    def __init__(self, edo: int = 12):
        self.edo = edo
        if edo == 12:
            self.fifths_distance = self.FIFTHS_DISTANCE_12
        else:
            self.fifths_distance = self._build_generalized_fifths(edo)

    def _build_generalized_fifths(self, edo: int) -> np.ndarray:
        """Build a generalized circle-of-fifths distance for arbitrary EDO."""
        fifth_steps = round(edo * 7 / 12)
        dist = np.zeros((edo, edo), dtype=float)
        for i in range(edo):
            for j in range(edo):
                diff = (j - i) % edo
                steps = 0
                current = 0
                while current != diff and steps < edo:
                    current = (current + fifth_steps) % edo
                    steps += 1
                dist[i, j] = steps
        return dist

    def chord_distance(self, pc_set_a: np.ndarray, root_a: int,
                       pc_set_b: np.ndarray, root_b: int) -> float:
        """
        Compute the tonal distance between two chords.

        Combines root distance on the circle of fifths with
        pitch-class set symmetric difference.
        """
        root_dist = self.fifths_distance[root_a % self.edo, root_b % self.edo]
        set_diff = np.sum(np.abs(pc_set_a - pc_set_b))
        return float(root_dist + 0.5 * set_diff)

    def key_distance(self, key_a: int, is_major_a: bool,
                     key_b: int, is_major_b: bool) -> float:
        """Compute distance between two keys."""
        base_dist = self.fifths_distance[key_a % self.edo, key_b % self.edo]
        mode_penalty = 0.0 if is_major_a == is_major_b else 1.0
        return float(base_dist + mode_penalty)


class MetricalGrid:
    """
    Constructs a hierarchical metrical grid for a given time signature.

    The grid assigns metrical weights to each sub-beat position,
    reflecting the cognitive salience of beats at different levels.
    """

    def __init__(self, beats_per_measure: int = 4,
                 steps_per_beat: int = 4,
                 num_measures: int = 4):
        self.beats_per_measure = beats_per_measure
        self.steps_per_beat = steps_per_beat
        self.num_measures = num_measures
        self.total_steps = beats_per_measure * steps_per_beat * num_measures
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute metrical weights for each step position."""
        weights = np.zeros(self.total_steps)
        for i in range(self.total_steps):
            level = 0
            if i % self.steps_per_beat == 0:
                level += 1  # beat level
            if i % (self.steps_per_beat * self.beats_per_measure) == 0:
                level += 1  # measure level
            if i % (self.steps_per_beat * self.beats_per_measure * 4) == 0:
                level += 1  # hyper-measure level
            if i == 0:
                level += 1  # downbeat of first measure
            weights[i] = level
        # Normalize to [0, 1]
        if weights.max() > 0:
            weights = weights / weights.max()
        return weights

    def get_weight(self, step: int) -> float:
        return float(self.weights[step % self.total_steps])


class GroupingAnalyzer:
    """
    Analyzes and generates grouping structure for musical passages.

    Implements GTTM Grouping Well-Formedness Rules (GWFRs) and
    Grouping Preference Rules (GPRs) to segment music into
    hierarchical groups (motives, phrases, sections).
    """

    # Common phrase lengths in progressive rock (in measures)
    PROG_ROCK_PHRASE_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
    JAZZ_FUSION_PHRASE_LENGTHS = [2, 4, 8, 12, 16, 32]

    def __init__(self, genre: str = "prog_rock"):
        self.genre = genre
        self.phrase_lengths = (
            self.PROG_ROCK_PHRASE_LENGTHS if genre == "prog_rock"
            else self.JAZZ_FUSION_PHRASE_LENGTHS
        )

    def generate_phrase_structure(
        self,
        total_measures: int,
        rng: np.random.Generator
    ) -> List[Dict]:
        """
        Generate a hierarchical phrase structure for a given number of measures.

        Returns a list of phrase descriptors with type, start, and length.
        """
        phrases = []
        current_measure = 0

        if self.genre == "prog_rock":
            sections = self._generate_prog_rock_form(total_measures, rng)
        else:
            sections = self._generate_jazz_fusion_form(total_measures, rng)

        for section in sections:
            phrases.append({
                "name": section["name"],
                "type": section["type"],
                "start": current_measure,
                "lgth": section["length"],
            })
            current_measure += section["length"]

        return phrases

    def _generate_prog_rock_form(
        self, total_measures: int, rng: np.random.Generator
    ) -> List[Dict]:
        """Generate a progressive rock song form."""
        sections = []
        remaining = total_measures

        # Typical prog rock form: Intro - Theme A - Theme B - Development - Solo - Recap - Coda
        form_template = [
            ("intro", "i", 0.08),
            ("theme_a", "A", 0.18),
            ("theme_b", "B", 0.15),
            ("development", "C", 0.20),
            ("solo", "S", 0.15),
            ("theme_a_reprise", "A", 0.14),
            ("coda", "o", 0.10),
        ]

        for name, stype, proportion in form_template:
            length = max(2, round(total_measures * proportion))
            # Snap to nearest valid phrase length
            valid = [l for l in self.phrase_lengths if l <= remaining]
            if not valid:
                break
            length = min(length, max(valid))
            sections.append({"name": name, "type": stype, "length": length})
            remaining -= length
            if remaining <= 0:
                break

        # Distribute any remaining measures
        if remaining > 0 and sections:
            sections[-1]["length"] += remaining

        return sections

    def _generate_jazz_fusion_form(
        self, total_measures: int, rng: np.random.Generator
    ) -> List[Dict]:
        """Generate a jazz fusion song form."""
        sections = []
        remaining = total_measures

        # Typical jazz fusion form: Head In - Solo 1 - Solo 2 - Interlude - Trading - Head Out
        form_template = [
            ("head_in", "A", 0.20),
            ("solo_1", "S", 0.20),
            ("interlude", "B", 0.10),
            ("solo_2", "S", 0.20),
            ("trading_fours", "T", 0.15),
            ("head_out", "A", 0.15),
        ]

        for name, stype, proportion in form_template:
            length = max(2, round(total_measures * proportion))
            valid = [l for l in self.phrase_lengths if l <= remaining]
            if not valid:
                break
            length = min(length, max(valid))
            sections.append({"name": name, "type": stype, "length": length})
            remaining -= length
            if remaining <= 0:
                break

        if remaining > 0 and sections:
            sections[-1]["length"] += remaining

        return sections


class TensionCurveGenerator:
    """
    Generates tension curves based on GTTM prolongational reduction.

    The tension curve defines the target affective trajectory of the piece,
    guiding the Schrödinger Bridge toward musically meaningful paths.
    """

    # Archetypal tension profiles for different section types
    TENSION_PROFILES = {
        "intro": lambda t: 0.2 + 0.3 * t,
        "theme_a": lambda t: 0.4 + 0.2 * np.sin(2 * np.pi * t),
        "theme_b": lambda t: 0.5 + 0.2 * np.sin(2 * np.pi * t + np.pi / 4),
        "development": lambda t: 0.3 + 0.5 * t,
        "solo": lambda t: 0.5 + 0.4 * np.sin(np.pi * t),
        "recapitulation": lambda t: 0.6 - 0.3 * t,
        "coda": lambda t: 0.5 * (1 - t),
        "head_in": lambda t: 0.3 + 0.3 * np.sin(np.pi * t),
        "head_out": lambda t: 0.5 - 0.3 * t,
        "interlude": lambda t: 0.3 + 0.1 * np.sin(4 * np.pi * t),
        "trading_fours": lambda t: 0.5 + 0.3 * np.sin(8 * np.pi * t),
        "solo_section": lambda t: 0.4 + 0.5 * np.sin(np.pi * t),
    }

    def __init__(self, resolution: int = 16):
        self.resolution = resolution

    def generate_curve(
        self,
        phrases: List[Dict],
        beats_per_measure: int = 4,
        steps_per_beat: int = 4
    ) -> np.ndarray:
        """
        Generate a tension curve for the entire piece.

        The curve is sampled at the specified resolution and smoothly
        interpolates between section-level tension profiles.
        """
        total_measures = sum(p["lgth"] for p in phrases)
        total_steps = total_measures * beats_per_measure * steps_per_beat
        curve = np.zeros(total_steps)

        current_step = 0
        for phrase in phrases:
            phrase_steps = phrase["lgth"] * beats_per_measure * steps_per_beat
            section_type = phrase.get("name", phrase.get("type", "theme_a"))

            # Find matching profile or use default
            profile_fn = self.TENSION_PROFILES.get(
                section_type,
                lambda t: 0.5 + 0.1 * np.sin(2 * np.pi * t)
            )

            t = np.linspace(0, 1, phrase_steps)
            curve[current_step:current_step + phrase_steps] = profile_fn(t)
            current_step += phrase_steps

        # Smooth the curve at section boundaries
        kernel_size = min(self.resolution, len(curve) // 4)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            curve = np.convolve(curve, kernel, mode="same")

        return np.clip(curve, 0.0, 1.0)


class GTTMPrior:
    """
    Combined GTTM structural prior.

    Integrates grouping, metrical, tonal distance, and tension features
    into a unified energy function for scoring musical states.
    """

    def __init__(self, config, edo: int = 12, genre: str = "prog_rock"):
        self.config = config
        self.tonal_metric = TonalDistanceMetric(edo=edo)
        self.grouping = GroupingAnalyzer(genre=genre)
        self.tension_gen = TensionCurveGenerator(
            resolution=config.tension_curve_resolution
        )

    def _extract_pitch_classes(self, state) -> tuple:
        """Extract pitch classes and bass from either BeatState or MusicalState."""
        if hasattr(state, 'pitch_classes'):
            return state.pitch_classes, state.bass_pitch_class
        elif hasattr(state, 'pitch_vector'):
            # Convert pitch_vector (128,) to pitch_class set (edo,)
            pc = np.zeros(self.tonal_metric.edo)
            for p in range(len(state.pitch_vector)):
                if state.pitch_vector[p] > 0.5:
                    pc[p % self.tonal_metric.edo] = 1.0
            bass = getattr(state, 'chord_root', 0)
            return pc, bass
        else:
            return np.zeros(self.tonal_metric.edo), 0

    def _extract_metrical_weight(self, state) -> float:
        """Extract metrical weight from either state type."""
        return getattr(state, 'metrical_weight', 0.5)

    def _extract_tension(self, state) -> float:
        """Extract tension from either state type."""
        return getattr(state, 'tension', 0.5)

    def _extract_grouping_level(self, state) -> int:
        """Extract grouping level from either state type."""
        return getattr(state, 'grouping_level', 0)

    def compute_transition_energy(
        self,
        state_a,
        state_b,
        target_tension: float
    ) -> float:
        """
        Compute the energy (negative log-probability) of transitioning
        from state_a to state_b, given a target tension level.

        Lower energy = more favorable transition.
        Accepts both BeatState and MusicalState objects.
        """
        # Tonal distance component
        pc_a, bass_a = self._extract_pitch_classes(state_a)
        pc_b, bass_b = self._extract_pitch_classes(state_b)
        tonal_dist = self.tonal_metric.chord_distance(pc_a, bass_a, pc_b, bass_b)

        # Metrical congruence: transitions on strong beats should be more significant
        metrical_factor = 1.0 + self._extract_metrical_weight(state_b)

        # Tension alignment: penalize deviation from target tension
        tension_error = abs(self._extract_tension(state_b) - target_tension)

        # Grouping coherence: penalize large jumps within a group
        grouping_penalty = 0.0
        if self._extract_grouping_level(state_a) == self._extract_grouping_level(state_b):
            grouping_penalty = max(0, tonal_dist - 3.0)

        energy = (
            self.config.grouping_weight * grouping_penalty
            + self.config.metrical_weight * (1.0 / metrical_factor)
            + self.config.time_span_weight * tonal_dist
            + self.config.prolongational_weight * tension_error
        )

        return energy

    def score_trajectory(
        self,
        trajectory: List[BeatState],
        tension_curve: np.ndarray
    ) -> float:
        """Score an entire trajectory against the GTTM prior."""
        total_energy = 0.0
        for i in range(len(trajectory) - 1):
            t_idx = min(i, len(tension_curve) - 1)
            total_energy += self.compute_transition_energy(
                trajectory[i], trajectory[i + 1], tension_curve[t_idx]
            )
        return total_energy
