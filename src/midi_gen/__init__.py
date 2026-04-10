"""
MIDI Generation Module for Progressive Rock and Jazz Fusion.

A GTTM-informed Schrödinger Bridge approach to symbolic music generation,
designed to integrate with the whole-song-gen cascaded diffusion pipeline.
"""

__version__ = "0.1.0"
__author__ = "SNET AI Music Team"

from midi_gen.core.pipeline import MidiGenerationPipeline
from midi_gen.core.config import GenerationConfig

__all__ = ["MidiGenerationPipeline", "GenerationConfig"]
