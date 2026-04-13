"""Training utilities for the candidate scorer.

Implements contrastive / energy-based learning from corpus data.
The training loop takes pairs of (context, positive_next, negative_next)
and trains the scorer to assign lower energy to the positive (corpus-observed)
transition than to the negative (randomly sampled or gated-but-unobserved)
transition.

This follows the revised strategy note (Section 4a, Step 4): "At first,
only ask it to rank plausible next states in context."
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ScorerTrainer:
    """Train the CandidateScorer on corpus data.

    Parameters
    ----------
    scorer : CandidateScorer
        The scorer model to train.
    learning_rate : float
        Learning rate for the optimizer.
    margin : float
        Margin for the contrastive loss (positive energy should be at least
        ``margin`` lower than negative energy).
    context_length : int
        Number of recent beats used as context.
    """

    def __init__(
        self,
        scorer,
        learning_rate: float = 1e-4,
        margin: float = 1.0,
        context_length: int = 16,
    ):
        self.scorer = scorer
        self.margin = margin
        self.context_length = context_length

        if HAS_TORCH and scorer.model is not None:
            self.optimizer = optim.Adam(scorer.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None

    def train_epoch(
        self,
        sequences: List[BeatLevelSequence],
        batch_size: int = 32,
        negative_samples: int = 8,
    ) -> float:
        """Train for one epoch over the given sequences.

        For each position in each sequence, the positive example is the
        actual next state, and negatives are randomly sampled states from
        the same sequence (but different positions).

        Parameters
        ----------
        sequences : list of BeatLevelSequence
            Corpus sequences to train on.
        batch_size : int
            Training batch size.
        negative_samples : int
            Number of negative examples per positive.

        Returns
        -------
        float
            Average loss for the epoch.
        """
        if not HAS_TORCH or self.scorer.model is None:
            return 0.0

        model = self.scorer.model
        model.train()

        # Build training pairs
        pairs = self._build_pairs(sequences, negative_samples)
        if not pairs:
            return 0.0

        np.random.shuffle(pairs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            loss = self._train_batch(batch)
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _build_pairs(
        self,
        sequences: List[BeatLevelSequence],
        negative_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build (context, positive, negative) triples from corpus sequences."""
        pairs = []
        ctx_len = self.context_length

        for seq in sequences:
            if len(seq) < ctx_len + 2:
                continue

            mat = seq.to_matrix()  # (N, 8)

            for i in range(ctx_len, len(seq) - 1):
                context = mat[i - ctx_len:i]  # (ctx_len, 8)
                positive = mat[i]  # (8,)

                # Sample negatives from other positions in the sequence
                neg_indices = np.random.choice(
                    [j for j in range(len(seq)) if abs(j - i) > 2],
                    size=min(negative_samples, max(1, len(seq) - 5)),
                    replace=True,
                )
                for ni in neg_indices:
                    negative = mat[ni]  # (8,)
                    pairs.append((context, positive, negative))

        return pairs

    def _train_batch(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> float:
        """Train on a single batch of (context, positive, negative) triples."""
        contexts = np.stack([b[0] for b in batch])  # (B, L, 8)
        positives = np.stack([b[1] for b in batch])  # (B, 8)
        negatives = np.stack([b[2] for b in batch])  # (B, 8)

        ctx_t = torch.from_numpy(contexts).long()
        pos_t = torch.from_numpy(positives).long()
        neg_t = torch.from_numpy(negatives).long()

        model = self.scorer.model

        # Forward pass
        pos_energy = model(ctx_t, pos_t)  # (B,)
        neg_energy = model(ctx_t, neg_t)  # (B,)

        # Contrastive margin loss: pos_energy should be lower than neg_energy
        loss = torch.clamp(pos_energy - neg_energy + self.margin, min=0.0).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        if HAS_TORCH and self.scorer.model is not None:
            torch.save({
                "model": self.scorer.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state."""
        if HAS_TORCH and self.scorer.model is not None:
            ckpt = torch.load(path, map_location="cpu")
            self.scorer.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
