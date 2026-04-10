"""
Adapter for whole-song-gen pipeline integration.

This module provides the interface between the MIDI generation module
and the existing whole-song-gen cascaded diffusion pipeline. It
translates the output of our GTTM+SB generation into the format
expected by the whole-song-gen inference pipeline.
"""

import numpy as np
import os
import sys
from typing import Optional, Dict, List


class WholeSongGenAdapter:
    """
    Adapter that bridges the MIDI generation module with the
    whole-song-gen cascaded diffusion pipeline.

    The adapter can operate in two modes:
    1. Form Provider: Generates form-level input for the whole-song-gen
       pipeline, which then handles counterpoint, lead sheet, and
       accompaniment generation.
    2. Full Override: Provides complete piano-roll output that bypasses
       the whole-song-gen cascade entirely.
    """

    def __init__(self, whole_song_gen_path: Optional[str] = None):
        """
        Initialize the adapter.

        Args:
            whole_song_gen_path: Path to the whole-song-gen repository.
                If provided, enables direct pipeline integration.
        """
        self.wsg_path = whole_song_gen_path
        self.wsg_available = False

        if whole_song_gen_path and os.path.isdir(whole_song_gen_path):
            sys.path.insert(0, whole_song_gen_path)
            try:
                from experiments.whole_song_gen import WholeSongGeneration
                self.wsg_available = True
                self._WholeSongGeneration = WholeSongGeneration
            except ImportError:
                pass

    def create_form_input(
        self,
        phrase_string: str,
        key: int = 0,
        is_major: bool = True
    ) -> Dict:
        """
        Create a form-level input for the whole-song-gen pipeline.

        This is the primary integration point: our module generates
        the structural plan, and whole-song-gen fills in the details.

        Args:
            phrase_string: Form string (e.g., "i4A8B8C12S8A8o4")
            key: Tonic pitch class (0-11)
            is_major: Whether the key is major

        Returns:
            Dictionary with form parameters for whole-song-gen
        """
        return {
            "pstring": phrase_string,
            "key": key,
            "is_major": is_major,
            "nbpm": 4,
            "nspb": 4,
        }

    def create_piano_roll_input(
        self,
        piano_roll: np.ndarray,
        level: str = "form"
    ) -> np.ndarray:
        """
        Format a piano-roll for injection at a specific level
        of the whole-song-gen cascade.

        Args:
            piano_roll: (2, T, 128) piano-roll array
            level: Which cascade level to inject at
                   ("form", "counterpoint", "leadsheet", "accompaniment")

        Returns:
            Formatted array for the specified level
        """
        if level == "form":
            # Form level expects (8, T_form) where T_form is downsampled
            return self._format_for_form(piano_roll)
        elif level == "counterpoint":
            # Counterpoint expects (2, T, 128)
            return piano_roll
        elif level == "leadsheet":
            return piano_roll
        elif level == "accompaniment":
            return piano_roll
        else:
            raise ValueError(f"Unknown level: {level}")

    def _format_for_form(self, piano_roll: np.ndarray) -> np.ndarray:
        """
        Downsample piano-roll to form-level representation.

        The form level operates at a coarser time resolution
        (one step per measure) and encodes phrase types and keys.
        """
        # Downsample: average over each measure (16 steps)
        steps_per_measure = 16  # 4 beats * 4 steps
        num_measures = piano_roll.shape[1] // steps_per_measure

        form = np.zeros((8, num_measures))
        for m in range(num_measures):
            start = m * steps_per_measure
            end = start + steps_per_measure
            measure_slice = piano_roll[:, start:end, :]

            # Encode pitch distribution
            pitch_dist = measure_slice[0].sum(axis=0)  # onset distribution
            if pitch_dist.sum() > 0:
                pitch_dist = pitch_dist / pitch_dist.sum()

            # Extract features for form representation
            form[0, m] = pitch_dist[48:].sum()  # melody activity
            form[1, m] = pitch_dist[:48].sum()   # bass activity
            form[2, m] = measure_slice[0].sum()   # onset density
            form[3, m] = measure_slice[1].sum()   # sustain density
            # Pitch centroid
            pitches = np.arange(128)
            if pitch_dist.sum() > 0:
                form[4, m] = np.average(pitches, weights=pitch_dist) / 128.0
            # Pitch spread
            form[5, m] = np.std(pitch_dist) if pitch_dist.sum() > 0 else 0
            # Rhythmic density
            form[6, m] = (measure_slice[0].sum(axis=1) > 0).mean()
            # Harmonic complexity (number of distinct pitch classes)
            active_pcs = set()
            for t in range(start, min(end, piano_roll.shape[1])):
                for p in range(128):
                    if piano_roll[0, t, p] > 0.5:
                        active_pcs.add(p % 12)
            form[7, m] = len(active_pcs) / 12.0

        return form

    def run_whole_song_gen(
        self,
        phrase_string: str,
        key: int = 0,
        is_major: bool = True,
        num_samples: int = 1,
        output_dir: str = "demo"
    ) -> Optional[str]:
        """
        Run the whole-song-gen pipeline with our form input.

        Requires the whole-song-gen repository to be available
        with pretrained models.
        """
        if not self.wsg_available:
            print("[WSG Adapter] whole-song-gen not available.")
            print("[WSG Adapter] To enable, provide the path to the repository:")
            print("[WSG Adapter]   adapter = WholeSongGenAdapter('/path/to/whole-song-gen')")
            return None

        try:
            import torch

            # Initialize the whole-song-gen pipeline
            wsg = self._WholeSongGeneration.init_pipeline(
                frm_model_folder='results_default/frm---/v-default',
                ctp_model_folder='results_default/ctp-a-b-/v-default',
                lsh_model_folder='results_default/lsh-a-b-/v-default',
                acc_model_folder='results_default/acc-a-b-/v-default',
            )

            wsg.main(
                n_sample=num_samples,
                nbpm=4,
                nspb=4,
                phrase_string=phrase_string,
                key=key,
                is_major=is_major,
                demo_dir=output_dir
            )

            return output_dir

        except Exception as e:
            print(f"[WSG Adapter] Error running whole-song-gen: {e}")
            return None

    def get_integration_status(self) -> Dict:
        """Return the current integration status."""
        return {
            "wsg_path": self.wsg_path,
            "wsg_available": self.wsg_available,
            "integration_modes": [
                "form_provider",
                "full_override" if self.wsg_available else None,
            ],
        }
