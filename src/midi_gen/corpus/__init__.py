"""Corpus ingestion and normalization pipeline.

Implements the gold / silver / bronze data tier system described in the
revised strategy note (Section 5):

    **Gold** — Native symbolic sources (MIDI, MusicXML, DAW exports).
              Used for training the candidate scorer and multitrack realizer.

    **Silver** — Paired symbolic + audio resources.
                Used for learning multitrack texture statistics, instrument
                role interaction, density envelopes, audio rendering.

    **Bronze** — Audio-transcribed material (source separation + AMT).
                Used for broad structural habits: groove families, density
                envelopes, rough chord movement, bar-level texture types.
                Should NOT dominate fine note-exact supervision.

The pipeline ingests data from any tier into a common event representation,
then derives beat-level and bar-level training examples.
"""

from midi_gen.corpus.ingestor import CorpusIngestor, DataTier
from midi_gen.corpus.normalizer import EventNormalizer

__all__ = ["CorpusIngestor", "DataTier", "EventNormalizer"]
