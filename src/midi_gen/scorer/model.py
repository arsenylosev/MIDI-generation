"""Candidate scorer model.

A compact transformer-based energy model that scores beat-level state
transitions.  Given the recent context (a window of beat-level states)
and a set of candidate next states, it outputs a scalar score for each
candidate indicating stylistic plausibility.

Architecture choices (from revised strategy note, Section 4a):
    - Input: context window of beat-level state vectors + one candidate
    - Embedding: per-field learned embeddings concatenated
    - Encoder: small causal transformer (2–4 layers, 128–256 dim)
    - Output: scalar energy (lower = more plausible)

The model can be used in two modes:
    1. **Ranking mode** (inference): score all candidates, pick the best
    2. **Training mode**: contrastive / energy-based learning from corpus
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence, VOCAB_SIZES

# Try to import torch; fall back to numpy-only scoring if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Field embedding dimensions
# ---------------------------------------------------------------------------

FIELD_NAMES = list(VOCAB_SIZES.keys())
N_FIELDS = len(FIELD_NAMES)


class CandidateScorer:
    """Score candidate next states given a context window.

    When PyTorch is available, uses a learned transformer model.
    Otherwise, falls back to a rule-based heuristic scorer that
    combines tonal distance, melodic smoothness, and groove consistency.

    Parameters
    ----------
    context_length : int
        Number of recent beats to use as context (default 16).
    embed_dim : int
        Embedding dimension per field (default 32).
    n_layers : int
        Number of transformer layers (default 2).
    n_heads : int
        Number of attention heads (default 4).
    genre : str
        Target genre for heuristic fallback scoring.
    """

    def __init__(
        self,
        context_length: int = 16,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        genre: str = "prog_rock",
    ):
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.genre = genre
        self.model = None

        if HAS_TORCH:
            self.model = _TransformerScorer(
                context_length=context_length,
                embed_dim=embed_dim,
                n_layers=n_layers,
                n_heads=n_heads,
            )

    def score(
        self,
        context: BeatLevelSequence,
        candidates: List[BeatLevelState],
    ) -> np.ndarray:
        """Score each candidate given the context.

        Parameters
        ----------
        context : BeatLevelSequence
            Recent beat-level states (will be truncated to context_length).
        candidates : list of BeatLevelState
            Candidate next states from the sparse generator.

        Returns
        -------
        np.ndarray
            Shape ``(len(candidates),)`` — lower scores are more plausible.
        """
        if self.model is not None and HAS_TORCH:
            return self._score_neural(context, candidates)
        else:
            return self._score_heuristic(context, candidates)

    def rank(
        self,
        context: BeatLevelSequence,
        candidates: List[BeatLevelState],
        top_k: int = 1,
    ) -> List[BeatLevelState]:
        """Return the top-k candidates ranked by score (best first)."""
        scores = self.score(context, candidates)
        indices = np.argsort(scores)[:top_k]
        return [candidates[i] for i in indices]

    def save(self, path: str) -> None:
        """Save model weights."""
        if self.model is not None and HAS_TORCH:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        if self.model is not None and HAS_TORCH:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model.eval()

    # --- Neural scoring ---

    def _score_neural(
        self,
        context: BeatLevelSequence,
        candidates: List[BeatLevelState],
    ) -> np.ndarray:
        """Score using the learned transformer model."""
        # Prepare context tensor: (1, context_length, N_FIELDS)
        ctx_mat = context.to_matrix()
        if len(ctx_mat) > self.context_length:
            ctx_mat = ctx_mat[-self.context_length:]
        elif len(ctx_mat) < self.context_length:
            pad = np.zeros((self.context_length - len(ctx_mat), 8), dtype=np.int32)
            ctx_mat = np.concatenate([pad, ctx_mat], axis=0)

        ctx_tensor = torch.from_numpy(ctx_mat).unsqueeze(0).long()  # (1, L, 8)

        # Prepare candidate tensors
        cand_vecs = np.stack([c.to_vector() for c in candidates])  # (K, 8)
        cand_tensor = torch.from_numpy(cand_vecs).long()  # (K, 8)

        # Score each candidate
        self.model.eval()
        with torch.no_grad():
            scores = []
            for k in range(len(candidates)):
                cand_k = cand_tensor[k].unsqueeze(0)  # (1, 8)
                energy = self.model(ctx_tensor, cand_k)  # scalar
                scores.append(energy.item())

        return np.array(scores, dtype=np.float32)

    # --- Heuristic scoring (fallback) ---

    def _score_heuristic(
        self,
        context: BeatLevelSequence,
        candidates: List[BeatLevelState],
    ) -> np.ndarray:
        """Rule-based heuristic scorer combining multiple musical criteria."""
        if len(context) == 0:
            return np.zeros(len(candidates), dtype=np.float32)

        current = context[-1]
        scores = np.zeros(len(candidates), dtype=np.float32)

        for i, cand in enumerate(candidates):
            score = 0.0

            # 1. Melodic smoothness (prefer stepwise motion)
            if current.melodic_head >= 0 and cand.melodic_head >= 0:
                interval = abs(cand.melodic_head - current.melodic_head)
                score += interval * 0.3  # penalize large leaps

            # 2. Chord distance (prefer close chords)
            if cand.chord_label != current.chord_label:
                from midi_gen.schema.beat_state import decode_chord
                r1, _ = decode_chord(current.chord_label)
                r2, _ = decode_chord(cand.chord_label)
                fifth_dist = self._fifth_distance(r1, r2)
                score += fifth_dist * 0.5

            # 3. Groove consistency (penalize changes)
            if cand.groove_token != current.groove_token:
                score += 1.0

            # 4. Boundary appropriateness
            if cand.boundary_level > 0 and current.beat_position != 0:
                score += 2.0  # boundaries should be on strong beats

            # 5. Voice leading: prefer common tones
            if cand.melodic_head >= 0 and current.melodic_head >= 0:
                if (cand.melodic_head % 12) == (current.melodic_head % 12):
                    score -= 0.5  # reward common tone

            scores[i] = score

        return scores

    @staticmethod
    def _fifth_distance(r1: int, r2: int) -> int:
        forward = 0
        r = r1
        while r != r2 and forward < 12:
            r = (r + 7) % 12
            forward += 1
        backward = 0
        r = r1
        while r != r2 and backward < 12:
            r = (r - 7) % 12
            backward += 1
        return min(forward, backward)


# ---------------------------------------------------------------------------
# PyTorch model (only defined if torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _FieldEmbedding(nn.Module):
        """Embed each field of a beat-level state vector separately."""

        def __init__(self, embed_dim: int = 32):
            super().__init__()
            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(vocab_size, embed_dim)
                for name, vocab_size in VOCAB_SIZES.items()
            })
            self.proj = nn.Linear(N_FIELDS * embed_dim, embed_dim * 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (..., N_FIELDS) int tensor → (..., embed_dim*4)."""
            parts = []
            for i, name in enumerate(FIELD_NAMES):
                parts.append(self.embeddings[name](x[..., i]))
            cat = torch.cat(parts, dim=-1)  # (..., N_FIELDS * embed_dim)
            return self.proj(cat)

    class _TransformerScorer(nn.Module):
        """Compact transformer that scores a candidate given context."""

        def __init__(
            self,
            context_length: int = 16,
            embed_dim: int = 32,
            n_layers: int = 2,
            n_heads: int = 4,
        ):
            super().__init__()
            hidden = embed_dim * 4
            self.field_embed = _FieldEmbedding(embed_dim)
            self.pos_embed = nn.Embedding(context_length + 1, hidden)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )
            self.energy_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(
            self,
            context: torch.Tensor,
            candidate: torch.Tensor,
        ) -> torch.Tensor:
            """
            context:   (B, L, 8) int tensor — recent beat states
            candidate: (B, 8) int tensor — one candidate next state

            Returns: (B,) float tensor — energy scores
            """
            B, L, _ = context.shape

            # Embed context and candidate
            ctx_emb = self.field_embed(context)  # (B, L, H)
            cand_emb = self.field_embed(candidate).unsqueeze(1)  # (B, 1, H)

            # Concatenate: [context..., candidate]
            seq = torch.cat([ctx_emb, cand_emb], dim=1)  # (B, L+1, H)

            # Add positional embeddings
            positions = torch.arange(L + 1, device=seq.device).unsqueeze(0)
            seq = seq + self.pos_embed(positions)

            # Transformer encoding
            out = self.transformer(seq)  # (B, L+1, H)

            # Take the last position (candidate) as the summary
            summary = out[:, -1, :]  # (B, H)

            # Energy head
            energy = self.energy_head(summary).squeeze(-1)  # (B,)
            return energy
