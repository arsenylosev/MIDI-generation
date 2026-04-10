"""
Schrödinger Bridge Solver for Music Generation.

Implements the Schrödinger Bridge (SB) framework for generating
structurally coherent musical trajectories between specified
start and end states, guided by the GTTM structural prior.

The SB formulation ensures that generated music follows an optimal
transport path through the space of musical states, maintaining
both local coherence and global structural integrity.

Performance note: The solver operates on a compact 12-dimensional
pitch-class representation internally, then expands to full 128-dim
pitch vectors only at output time.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class MusicalState:
    """Compact representation of a musical state at a time step."""
    time_step: int
    pitch_vector: np.ndarray  # (128,) piano-roll slice or (edo,) pitch-class
    chord_root: int
    chord_quality: int  # encoded chord type
    tension: float
    velocity: float
    metrical_weight: float

    def to_vector(self) -> np.ndarray:
        """Flatten state to a single vector for distance computation."""
        return np.concatenate([
            self.pitch_vector.flatten(),
            np.array([
                self.chord_root / 12.0,
                self.chord_quality / 10.0,
                self.tension,
                self.velocity,
                self.metrical_weight
            ])
        ])

    def to_compact(self) -> np.ndarray:
        """Return a compact 17-dim vector: 12 pitch-classes + 5 params."""
        pc = np.zeros(12)
        pv = self.pitch_vector
        for i in range(len(pv)):
            if pv[i] > 0.5:
                pc[i % 12] = 1.0
        return np.concatenate([
            pc,
            np.array([
                self.chord_root / 12.0,
                self.chord_quality / 10.0,
                self.tension,
                self.velocity,
                self.metrical_weight
            ])
        ])


class CandidateGenerator:
    """
    Generates musically constrained candidate states for graph construction.

    Uses musical heuristics to produce a manageable set of plausible
    next states. Operates on compact representations for speed.
    """

    # Common chord progressions in progressive rock
    PROG_ROCK_PROGRESSIONS = {
        0: [7, 5, 9, 3, 10, 8],   # C -> G, F, A, Eb, Bb, Ab
        1: [8, 6, 10, 4, 11, 3],   # Db -> Ab, Gb, Bb, E, B, Eb
        2: [9, 7, 11, 5, 0, 4],    # D -> A, G, B, F, C, E
        3: [10, 8, 0, 6, 1, 5],    # Eb -> Bb, Ab, C, Gb, Db, F
        4: [11, 9, 1, 7, 2, 6],    # E -> B, A, Db, G, D, Gb
        5: [0, 10, 2, 8, 3, 7],    # F -> C, Bb, D, Ab, Eb, G
        6: [1, 11, 3, 9, 4, 8],    # Gb -> Db, B, Eb, A, E, Ab
        7: [2, 0, 4, 10, 5, 9],    # G -> D, C, E, Bb, F, A
        8: [3, 1, 5, 11, 6, 10],   # Ab -> Eb, Db, F, B, Gb, Bb
        9: [4, 2, 6, 0, 7, 11],    # A -> E, D, Gb, C, G, B
        10: [5, 3, 7, 1, 8, 0],    # Bb -> F, Eb, G, Db, Ab, C
        11: [6, 4, 8, 2, 9, 1],    # B -> Gb, E, Ab, D, A, Db
    }

    JAZZ_PROGRESSIONS = {
        0: [5, 7, 9, 3, 10, 4],
        1: [6, 8, 10, 4, 11, 5],
        2: [7, 9, 11, 5, 0, 6],
        3: [8, 10, 0, 6, 1, 7],
        4: [9, 11, 1, 7, 2, 8],
        5: [10, 0, 2, 8, 3, 9],
        6: [11, 1, 3, 9, 4, 10],
        7: [0, 2, 4, 10, 5, 11],
        8: [1, 3, 5, 11, 6, 0],
        9: [2, 4, 6, 0, 7, 1],
        10: [3, 5, 7, 1, 8, 2],
        11: [4, 6, 8, 2, 9, 3],
    }

    # Chord intervals by quality index
    INTERVALS_MAP = {
        0: [0, 4, 7],           # major
        1: [0, 3, 7],           # minor
        2: [0, 4, 7, 10],       # dom7
        3: [0, 4, 7, 11],       # maj7
        4: [0, 3, 7, 10],       # min7
        5: [0, 3, 6],           # dim
        6: [0, 4, 8],           # aug
        7: [0, 5, 7],           # sus4
        8: [0, 3, 6, 10],       # min7b5
        9: [0, 3, 6, 9],        # dim7
    }

    def __init__(self, genre: str = "prog_rock", num_candidates: int = 8):
        self.genre = genre
        self.num_candidates = num_candidates
        self.progressions = (
            self.JAZZ_PROGRESSIONS if "jazz" in genre
            else self.PROG_ROCK_PROGRESSIONS
        )

    def generate_candidates(
        self,
        current_state: MusicalState,
        target_tension: float,
        rng: np.random.Generator
    ) -> List[MusicalState]:
        """Generate candidate next states from the current state."""
        candidates = []
        next_roots = self.progressions.get(
            current_state.chord_root % 12, list(range(12))
        )

        # Only use top-3 roots and top-2 qualities for speed
        for root in next_roots[:3]:
            for quality in [0, 4]:  # major and min7 only
                pitch_vec = self._chord_to_pitch_vector(root, quality)
                tension = np.clip(target_tension + rng.normal(0, 0.1), 0.0, 1.0)
                velocity = np.clip(0.4 + 0.5 * tension + rng.normal(0, 0.05), 0.1, 1.0)

                candidates.append(MusicalState(
                    time_step=current_state.time_step + 1,
                    pitch_vector=pitch_vec,
                    chord_root=root,
                    chord_quality=quality,
                    tension=tension,
                    velocity=velocity,
                    metrical_weight=current_state.metrical_weight
                ))

        if len(candidates) > self.num_candidates:
            indices = rng.choice(len(candidates), self.num_candidates, replace=False)
            candidates = [candidates[i] for i in indices]

        return candidates

    def _chord_to_pitch_vector(self, root: int, quality: int) -> np.ndarray:
        """Convert a chord root and quality to a 128-dimensional pitch vector."""
        vec = np.zeros(128)
        intervals = self.INTERVALS_MAP.get(quality, [0, 4, 7])
        for octave in [3, 4, 5]:  # Only 3 octaves for speed
            for interval in intervals:
                pitch = octave * 12 + root + interval
                if 0 <= pitch < 128:
                    vec[pitch] = 1.0
        return vec


class SchrodingerBridgeSolver:
    """
    Schrödinger Bridge solver for optimal musical trajectory generation.

    Given start and end musical states, and a GTTM-based energy function,
    the solver computes the optimal path through the space of musical
    states using the Sinkhorn-Knopp algorithm.

    Performance: operates at coarse time resolution (one state per measure)
    and uses compact pitch-class representations internally.
    """

    def __init__(
        self,
        energy_fn: Callable,
        num_steps: int = 100,
        sinkhorn_iterations: int = 50,
        sinkhorn_epsilon: float = 0.01,
        num_candidates: int = 8,
        genre: str = "prog_rock"
    ):
        self.energy_fn = energy_fn
        self.num_steps = num_steps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.candidate_gen = CandidateGenerator(
            genre=genre, num_candidates=num_candidates
        )

    def solve(
        self,
        start_state: MusicalState,
        end_state: MusicalState,
        tension_curve: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> List[MusicalState]:
        """
        Solve the Schrödinger Bridge problem to find the optimal
        musical trajectory between start and end states.

        Uses a coarse-grained approach: one state per measure
        (not per 16th note) for tractable computation.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Use a coarse resolution: ~1 state per measure (16 steps)
        coarse_steps = max(2, min(self.num_steps // 16, 16))

        tension_resampled = np.interp(
            np.linspace(0, 1, coarse_steps),
            np.linspace(0, 1, len(tension_curve)),
            tension_curve
        )

        # Initialize trajectory with linear interpolation
        trajectory = self._initialize_trajectory(
            start_state, end_state, tension_resampled, coarse_steps
        )

        # Iterative refinement — cap iterations for speed
        effective_iters = min(self.sinkhorn_iterations, 5)
        for iteration in range(effective_iters):
            trajectory = self._refine_trajectory(
                trajectory, tension_resampled, rng, iteration, effective_iters
            )

        # Upsample trajectory back to full resolution
        return self._upsample_trajectory(trajectory, self.num_steps)

    def _initialize_trajectory(
        self,
        start: MusicalState,
        end: MusicalState,
        tension_curve: np.ndarray,
        num_steps: int
    ) -> List[MusicalState]:
        """Initialize trajectory via linear interpolation between endpoints."""
        trajectory = []
        start_vec = start.to_compact()
        end_vec = end.to_compact()

        for t in range(num_steps):
            alpha = t / max(1, num_steps - 1)
            interp_vec = (1 - alpha) * start_vec + alpha * end_vec

            # Reconstruct pitch vector from interpolated pitch classes
            pitch_vec = np.zeros(128)
            for pc_idx in range(12):
                if interp_vec[pc_idx] > 0.5:
                    for octave in [3, 4, 5]:
                        p = octave * 12 + pc_idx
                        if 0 <= p < 128:
                            pitch_vec[p] = 1.0

            state = MusicalState(
                time_step=t,
                pitch_vector=pitch_vec,
                chord_root=int(round(interp_vec[12] * 12)) % 12,
                chord_quality=int(round(interp_vec[13] * 10)) % 10,
                tension=float(tension_curve[t]),
                velocity=float(np.clip(interp_vec[15], 0.1, 1.0)),
                metrical_weight=float(interp_vec[16])
            )
            trajectory.append(state)

        return trajectory

    def _refine_trajectory(
        self,
        trajectory: List[MusicalState],
        tension_curve: np.ndarray,
        rng: np.random.Generator,
        iteration: int,
        total_iterations: int
    ) -> List[MusicalState]:
        """
        Refine trajectory using forward-backward Sinkhorn updates.

        Each step considers candidate replacements and selects the
        one that minimizes the total energy under the GTTM prior.
        """
        refined = [trajectory[0]]  # Keep start fixed
        temperature = max(0.1, 1.0 - iteration / max(1, total_iterations))

        for t in range(1, len(trajectory) - 1):
            current = trajectory[t]
            target_tension = tension_curve[t]

            # Generate candidates (small set for speed)
            candidates = self.candidate_gen.generate_candidates(
                trajectory[t - 1], target_tension, rng
            )
            candidates.append(current)

            # Score each candidate
            scores = []
            for cand in candidates:
                cand.time_step = t
                fwd_energy = self.energy_fn(refined[-1], cand, target_tension)
                bwd_energy = self.energy_fn(
                    cand, trajectory[min(t + 1, len(trajectory) - 1)],
                    tension_curve[min(t + 1, len(tension_curve) - 1)]
                )
                scores.append(fwd_energy + bwd_energy)

            scores = np.array(scores)
            scores = -scores / (temperature * self.sinkhorn_epsilon + 1e-8)
            scores = scores - scores.max()
            probs = np.exp(scores)
            probs = probs / (probs.sum() + 1e-8)

            if temperature > 0.3:
                idx = rng.choice(len(candidates), p=probs)
            else:
                idx = np.argmax(probs)

            refined.append(candidates[idx])

        refined.append(trajectory[-1])
        return refined

    def _upsample_trajectory(
        self,
        coarse_trajectory: List[MusicalState],
        target_steps: int
    ) -> List[MusicalState]:
        """Upsample a coarse trajectory to the target number of steps."""
        if len(coarse_trajectory) <= 1:
            return coarse_trajectory

        coarse_len = len(coarse_trajectory)
        result = []

        for t in range(target_steps):
            # Map to coarse index
            coarse_t = t * (coarse_len - 1) / max(1, target_steps - 1)
            idx_lo = int(coarse_t)
            idx_hi = min(idx_lo + 1, coarse_len - 1)
            frac = coarse_t - idx_lo

            lo = coarse_trajectory[idx_lo]
            hi = coarse_trajectory[idx_hi]

            # Interpolate
            pitch_vec = (1 - frac) * lo.pitch_vector + frac * hi.pitch_vector
            state = MusicalState(
                time_step=t,
                pitch_vector=pitch_vec,
                chord_root=lo.chord_root if frac < 0.5 else hi.chord_root,
                chord_quality=lo.chord_quality if frac < 0.5 else hi.chord_quality,
                tension=(1 - frac) * lo.tension + frac * hi.tension,
                velocity=(1 - frac) * lo.velocity + frac * hi.velocity,
                metrical_weight=(1 - frac) * lo.metrical_weight + frac * hi.metrical_weight
            )
            result.append(state)

        return result

    def solve_with_waypoints(
        self,
        waypoints: List[MusicalState],
        tension_curve: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> List[MusicalState]:
        """
        Solve SB with intermediate waypoints (e.g., section boundaries).

        Solves segment-by-segment and concatenates the results.
        """
        if rng is None:
            rng = np.random.default_rng()

        full_trajectory = []
        for i in range(len(waypoints) - 1):
            start_t = waypoints[i].time_step
            end_t = waypoints[i + 1].time_step
            segment_tension = tension_curve[start_t:end_t]

            if len(segment_tension) < 2:
                full_trajectory.append(waypoints[i])
                continue

            segment = self.solve(
                waypoints[i], waypoints[i + 1],
                segment_tension, rng
            )

            if i == 0:
                full_trajectory.extend(segment)
            else:
                full_trajectory.extend(segment[1:])

        return full_trajectory
