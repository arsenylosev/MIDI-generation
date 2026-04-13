"""Corpus ingestor: loads MIDI / MusicXML files and converts them to
the native beat-level representation.

Supports the three data tiers (gold, silver, bronze) with different
levels of trust and different processing pipelines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    METER_MAP,
    encode_chord,
)

logger = logging.getLogger(__name__)


class DataTier(Enum):
    """Data quality tier."""
    GOLD = "gold"       # Native symbolic sources
    SILVER = "silver"   # Paired symbolic + audio
    BRONZE = "bronze"   # Audio-transcribed


@dataclass
class CorpusEntry:
    """A single corpus entry with metadata."""
    path: str
    tier: DataTier
    genre: str = ""
    artist: str = ""
    title: str = ""
    sequence: Optional[BeatLevelSequence] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorpusIngestor:
    """Ingest MIDI files into the native beat-level representation.

    Parameters
    ----------
    default_genre : str
        Default genre label for ingested files.
    quantize_resolution : int
        Quantization resolution in ticks per beat.
    """

    def __init__(
        self,
        default_genre: str = "prog_rock",
        quantize_resolution: int = 4,
    ):
        self.default_genre = default_genre
        self.quantize_resolution = quantize_resolution

    def ingest_file(
        self,
        path: str,
        tier: DataTier = DataTier.GOLD,
        genre: Optional[str] = None,
    ) -> Optional[CorpusEntry]:
        """Ingest a single MIDI file.

        Parameters
        ----------
        path : str
            Path to the MIDI file.
        tier : DataTier
            Quality tier for this file.
        genre : str, optional
            Genre label (defaults to self.default_genre).

        Returns
        -------
        CorpusEntry or None
            The ingested entry, or None if parsing failed.
        """
        if pretty_midi is None:
            logger.warning("pretty_midi not installed; cannot ingest MIDI files")
            return None

        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            return None

        genre = genre or self.default_genre
        sequence = self._midi_to_sequence(pm, genre)

        return CorpusEntry(
            path=path,
            tier=tier,
            genre=genre,
            sequence=sequence,
            metadata={
                "duration_sec": pm.get_end_time(),
                "n_instruments": len(pm.instruments),
                "tempo": float(pm.get_tempo_changes()[1][0]) if len(pm.get_tempo_changes()[1]) > 0 else 120.0,
            },
        )

    def ingest_directory(
        self,
        directory: str,
        tier: DataTier = DataTier.GOLD,
        genre: Optional[str] = None,
        extensions: tuple = (".mid", ".midi"),
    ) -> List[CorpusEntry]:
        """Ingest all MIDI files in a directory.

        Parameters
        ----------
        directory : str
            Path to the directory.
        tier : DataTier
            Quality tier for all files in this directory.
        genre : str, optional
            Genre label.
        extensions : tuple
            File extensions to include.

        Returns
        -------
        list of CorpusEntry
            Successfully ingested entries.
        """
        entries = []
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return entries

        for fpath in sorted(dir_path.rglob("*")):
            if fpath.suffix.lower() in extensions:
                entry = self.ingest_file(str(fpath), tier=tier, genre=genre)
                if entry is not None:
                    entries.append(entry)

        logger.info(f"Ingested {len(entries)} files from {directory} (tier={tier.value})")
        return entries

    def save_corpus(self, entries: List[CorpusEntry], output_dir: str) -> None:
        """Save ingested corpus to disk as JSON sequences.

        Parameters
        ----------
        entries : list of CorpusEntry
            Ingested corpus entries.
        output_dir : str
            Output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        manifest = []
        for i, entry in enumerate(entries):
            if entry.sequence is not None:
                seq_path = out / f"{i:05d}_{Path(entry.path).stem}.json"
                entry.sequence.to_json(str(seq_path))
                manifest.append({
                    "index": i,
                    "path": entry.path,
                    "tier": entry.tier.value,
                    "genre": entry.genre,
                    "seq_file": seq_path.name,
                    "n_beats": len(entry.sequence),
                    **entry.metadata,
                })

        manifest_path = out / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved {len(manifest)} sequences to {output_dir}")

    def load_corpus(self, corpus_dir: str) -> List[BeatLevelSequence]:
        """Load a saved corpus from disk.

        Parameters
        ----------
        corpus_dir : str
            Directory containing the saved corpus.

        Returns
        -------
        list of BeatLevelSequence
            Loaded sequences.
        """
        dir_path = Path(corpus_dir)
        manifest_path = dir_path / "manifest.json"

        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return []

        with open(manifest_path) as f:
            manifest = json.load(f)

        sequences = []
        for entry in manifest:
            seq_path = dir_path / entry["seq_file"]
            if seq_path.exists():
                seq = BeatLevelSequence.from_json(str(seq_path))
                sequences.append(seq)

        logger.info(f"Loaded {len(sequences)} sequences from {corpus_dir}")
        return sequences

    # --- Internal conversion ---

    def _midi_to_sequence(
        self,
        pm: "pretty_midi.PrettyMIDI",
        genre: str,
    ) -> BeatLevelSequence:
        """Convert a PrettyMIDI object to a BeatLevelSequence.

        This is a simplified conversion that:
        - Quantizes to beat boundaries
        - Estimates chords from simultaneous notes
        - Detects phrase boundaries from gaps
        - Assigns default meter and groove tokens
        """
        # Use tempo from the MIDI file; fall back to 120 BPM
        tempo_changes = pm.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            bpm = float(tempo_changes[1][0])
        else:
            bpm = 120.0
        seconds_per_beat = 60.0 / bpm
        total_beats = int(pm.get_end_time() / seconds_per_beat) + 1

        states = []
        for beat_idx in range(total_beats):
            t = beat_idx * seconds_per_beat

            # Collect all active notes at this beat
            active_pitches = []
            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                for note in inst.notes:
                    if note.start <= t < note.end:
                        active_pitches.append(note.pitch)

            # Estimate melodic head (highest pitch)
            melodic_head = max(active_pitches) if active_pitches else -1

            # Estimate chord (simplified: most common pitch class)
            if active_pitches:
                pcs = [p % 12 for p in active_pitches]
                root = max(set(pcs), key=pcs.count)
                # Simple quality detection
                pc_set = set(pcs)
                if (root + 4) % 12 in pc_set and (root + 7) % 12 in pc_set:
                    if (root + 10) % 12 in pc_set:
                        quality = 2  # dom7
                    elif (root + 11) % 12 in pc_set:
                        quality = 3  # maj7
                    else:
                        quality = 0  # maj
                elif (root + 3) % 12 in pc_set and (root + 7) % 12 in pc_set:
                    if (root + 10) % 12 in pc_set:
                        quality = 4  # min7
                    else:
                        quality = 1  # min
                else:
                    quality = 0  # default to major
                chord_label = encode_chord(root, quality)
            else:
                chord_label = 0

            # Detect boundaries from note density changes
            boundary_level = 0
            if beat_idx > 0 and not active_pitches:
                boundary_level = 1  # gap → sub-phrase boundary
            if beat_idx % 16 == 0 and beat_idx > 0:
                boundary_level = max(boundary_level, 2)  # every 4 bars → phrase
            if beat_idx % 64 == 0 and beat_idx > 0:
                boundary_level = 3  # every 16 bars → section

            states.append(BeatLevelState(
                meter_token=0,  # default 4/4
                beat_position=beat_idx % 4,
                boundary_level=boundary_level,
                region_label=beat_idx // 64,
                chord_label=chord_label,
                harmonic_role=0,
                melodic_head=melodic_head,
                groove_token=0,
            ))

        return BeatLevelSequence(
            states=states,
            bpm=bpm,
            genre=genre,
        )
