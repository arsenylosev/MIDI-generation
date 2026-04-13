"""Structural evaluation framework.

Implements the evaluation metrics from the revised strategy note (Section 11):

    Early evaluation should not fixate on generic symbolic music metrics.
    The first question is whether the music *moves* in the way the project
    wants — whether generated pieces exhibit purposeful structural behavior.

Evaluation targets:
    - Phrase boundary clarity
    - Cadence arrival quality
    - Repetition with variation
    - Meter-sensitive harmonic rhythm
    - Multitrack role differentiation
    - Subjective sense of directed movement
"""

from midi_gen.evaluation.structural_metrics import StructuralEvaluator
from midi_gen.evaluation.challenge_sets import ChallengeSetGenerator

__all__ = ["StructuralEvaluator", "ChallengeSetGenerator"]
