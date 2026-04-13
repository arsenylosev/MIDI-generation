"""
MIDI Generation Module for Progressive Rock and Jazz Fusion.

A GTTM-informed Schrödinger Bridge approach to symbolic music generation,
designed to integrate with the whole-song-gen cascaded diffusion pipeline.

v0.2.0 — Revised architecture with beat-level structural tokens, sparse
candidate generation, learned scoring, multitrack realization, guide-audio
rendering, and structural evaluation.
"""

__version__ = "0.2.0"
__author__ = "SNET AI Music Team"

from midi_gen.core.pipeline import MidiGenerationPipeline
from midi_gen.core.pipeline_v2 import MidiGenerationPipelineV2
from midi_gen.core.config import GenerationConfig

__all__ = [
    "MidiGenerationPipeline",
    "MidiGenerationPipelineV2",
    "GenerationConfig",
]
