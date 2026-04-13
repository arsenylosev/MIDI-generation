"""Learned candidate scorer / reranker.

Implements the neural component described in the revised strategy note
(Section 4a): a compact transformer-based model that ranks which
structurally allowed next states are most stylistically convincing
in context.

The scorer operates over the sparse candidates produced by the
SparseCandidateGenerator — it chooses among interpretable candidates
rather than hallucinating the whole future state space.
"""

from midi_gen.scorer.model import CandidateScorer
from midi_gen.scorer.training import ScorerTrainer

__all__ = ["CandidateScorer", "ScorerTrainer"]
