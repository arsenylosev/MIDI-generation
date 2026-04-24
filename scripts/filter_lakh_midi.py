#!/usr/bin/env python3
"""Gold-tier ingestion: Filter the Lakh MIDI Dataset by genre.

This script filters the LMD-matched subset using Million Song Dataset
metadata to isolate MIDI files tagged with progressive rock, jazz fusion,
and related sub-genres.  The filtered files are then converted to the
project's native BeatLevelSequence representation via the existing
``CorpusIngestor``.

Prerequisites
-------------
1. Download **LMD-matched** from https://colinraffel.com/projects/lmd/
   and extract to ``data/raw/lmd_matched/``.
2. Download **LMD-matched metadata** (HDF5 files) from the same page
   and extract to ``data/raw/lmd_matched_h5/``.
3. (Optional) Download the **MSD AllMusic genre annotations** from
   http://millionsongdataset.com/sites/default/files/AdditionalFiles/
   or use the tagtraum genre annotations.

Usage
-----
    python scripts/filter_lakh_midi.py \\
        --lmd-dir     data/raw/lmd_matched/ \\
        --metadata-dir data/raw/lmd_matched_h5/ \\
        --output-dir  data/gold/lakh_filtered/ \\
        --genre-file  data/raw/msd_tagtraum_cd2.cls

Dependencies
------------
    pip install pretty_midi tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target genre tags (case-insensitive matching)
# ---------------------------------------------------------------------------
# These tags are drawn from common MSD / Last.fm / AllMusic genre taxonomies.
# The list is intentionally broad to maximise recall; manual curation should
# follow to promote the best files to Gold and demote the rest to Silver.

TARGET_TAGS = {
    # Progressive rock
    "progressive rock", "prog rock", "prog", "progressive",
    "progressive metal", "prog metal", "art rock", "symphonic rock",
    "canterbury scene", "krautrock", "neo-prog", "post-progressive",
    "zeuhl", "rock in opposition",
    # Jazz fusion
    "jazz fusion", "fusion", "jazz-rock", "jazz rock",
    "jazz-funk", "jazz funk", "smooth jazz", "contemporary jazz",
    "avant-garde jazz", "electric jazz", "crossover jazz",
    # Adjacent styles that often share structural properties
    "math rock", "post-rock", "experimental rock",
    "jazz", "bebop", "hard bop", "modal jazz", "free jazz",
}


def _normalise_tag(tag: str) -> str:
    return tag.strip().lower()


# ---------------------------------------------------------------------------
# Genre file parsers
# ---------------------------------------------------------------------------

def load_tagtraum_genres(genre_file: str | Path) -> dict[str, set[str]]:
    """Load the tagtraum CD2 genre annotations.

    Format: ``<MSD_track_id>\\t<genre>`` (one per line).
    Returns a mapping from MSD track ID → set of genre strings.
    """
    mapping: dict[str, set[str]] = {}
    with open(genre_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                track_id = parts[0].strip()
                genre = _normalise_tag(parts[1])
                mapping.setdefault(track_id, set()).add(genre)
    log.info("Loaded genre annotations for %d tracks", len(mapping))
    return mapping


def load_lastfm_tags(tags_file: str | Path) -> dict[str, set[str]]:
    """Load Last.fm tags from the MSD SQLite dump or a TSV export.

    Expected TSV format: ``<MSD_track_id>\\t<tag>\\t<count>``
    """
    mapping: dict[str, set[str]] = {}
    with open(tags_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                track_id = row[0].strip()
                tag = _normalise_tag(row[1])
                mapping.setdefault(track_id, set()).add(tag)
    log.info("Loaded Last.fm tags for %d tracks", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# LMD-matched directory walker
# ---------------------------------------------------------------------------

def discover_lmd_files(lmd_dir: str | Path) -> dict[str, list[Path]]:
    """Walk the LMD-matched directory tree and group MIDI files by MSD ID.

    LMD-matched is organised as ``<lmd_dir>/<letter>/<letter>/<letter>/
    <MSD_TRACK_ID>/<hash>.mid``.
    """
    lmd_dir = Path(lmd_dir)
    msd_to_files: dict[str, list[Path]] = {}
    for midi_path in lmd_dir.rglob("*.mid"):
        # The parent directory name is the MSD track ID
        msd_id = midi_path.parent.name
        if msd_id.startswith("TR"):
            msd_to_files.setdefault(msd_id, []).append(midi_path)
    log.info(
        "Discovered %d MIDI files across %d MSD entries",
        sum(len(v) for v in msd_to_files.values()),
        len(msd_to_files),
    )
    return msd_to_files


# ---------------------------------------------------------------------------
# Quality pre-filter (lightweight, no heavy deps)
# ---------------------------------------------------------------------------

def _quick_midi_check(midi_path: Path) -> bool:
    """Return True if the MIDI file passes basic sanity checks."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        # Reject very short files (< 30 seconds)
        if pm.get_end_time() < 30.0:
            return False
        # Reject files with no notes
        total_notes = sum(len(inst.notes) for inst in pm.instruments)
        if total_notes < 50:
            return False
        # Reject files with only a single instrument
        non_drum = [i for i in pm.instruments if not i.is_drum]
        if len(non_drum) < 2:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main filter logic
# ---------------------------------------------------------------------------

def filter_lmd(
    lmd_dir: str | Path,
    genre_mapping: dict[str, set[str]],
    output_dir: str | Path,
    quality_check: bool = True,
) -> dict:
    """Filter LMD-matched by genre and copy matching files to output_dir.

    Returns a summary dict with counts and the list of matched MSD IDs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msd_to_files = discover_lmd_files(lmd_dir)

    matched_ids: list[str] = []
    copied_files = 0
    skipped_quality = 0

    for msd_id, midi_files in sorted(msd_to_files.items()):
        tags = genre_mapping.get(msd_id, set())
        matching_tags = tags & TARGET_TAGS
        if not matching_tags:
            continue

        # Copy the first (or best) MIDI file for this track
        midi_path = midi_files[0]
        if quality_check and not _quick_midi_check(midi_path):
            skipped_quality += 1
            continue

        dest_dir = output_dir / msd_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / midi_path.name
        shutil.copy2(midi_path, dest_path)

        # Write a sidecar metadata file
        meta = {
            "msd_id": msd_id,
            "source_file": midi_path.name,
            "matched_tags": sorted(matching_tags),
            "all_tags": sorted(tags),
            "tier": "gold",
        }
        with open(dest_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        matched_ids.append(msd_id)
        copied_files += 1

        if copied_files % 100 == 0:
            log.info("Copied %d files so far...", copied_files)

    summary = {
        "total_msd_entries": len(msd_to_files),
        "genre_matched": len(matched_ids),
        "quality_skipped": skipped_quality,
        "files_copied": copied_files,
        "target_tags": sorted(TARGET_TAGS),
        "matched_msd_ids": matched_ids,
    }

    summary_path = output_dir / "filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "Filtering complete: %d genre-matched, %d quality-skipped, %d copied",
        len(matched_ids), skipped_quality, copied_files,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Lakh MIDI Dataset by progressive rock / jazz fusion genre tags."
    )
    parser.add_argument(
        "--lmd-dir", required=True,
        help="Path to the extracted LMD-matched directory",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write filtered MIDI files",
    )
    parser.add_argument(
        "--genre-file", required=True,
        help="Path to genre annotation file (tagtraum .cls or Last.fm .tsv)",
    )
    parser.add_argument(
        "--genre-format", default="tagtraum",
        choices=["tagtraum", "lastfm"],
        help="Format of the genre annotation file",
    )
    parser.add_argument(
        "--no-quality-check", action="store_true",
        help="Skip the MIDI quality pre-filter",
    )
    args = parser.parse_args()

    if args.genre_format == "tagtraum":
        genre_mapping = load_tagtraum_genres(args.genre_file)
    else:
        genre_mapping = load_lastfm_tags(args.genre_file)

    filter_lmd(
        lmd_dir=args.lmd_dir,
        genre_mapping=genre_mapping,
        output_dir=args.output_dir,
        quality_check=not args.no_quality_check,
    )


if __name__ == "__main__":
    main()
