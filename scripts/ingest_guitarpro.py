#!/usr/bin/env python3
"""Gold-tier ingestion: GuitarPro (.gp3/.gp4/.gp5) → BeatLevelSequence JSON.

This script parses GuitarPro tablature files using the ``pyguitarpro`` library
and converts each file into the project's native beat-level representation.
GuitarPro files are considered the richest Gold-tier source because they carry
multi-track instrument separation (drums, bass, guitars, keys) that maps
directly onto the multitrack realizer's training needs.

Usage
-----
    python scripts/ingest_guitarpro.py \\
        --input-dir  data/raw/guitarpro/ \\
        --output-dir data/gold/guitarpro/ \\
        --tier gold

Dependencies
------------
    pip install pyguitarpro pretty_midi
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    import guitarpro
except ImportError:
    sys.exit(
        "pyguitarpro is required.  Install with:  pip install pyguitarpro"
    )

try:
    import pretty_midi
except ImportError:
    sys.exit("pretty_midi is required.  Install with:  pip install pretty_midi")

# ---------------------------------------------------------------------------
# Add project root to path so we can import the schema
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from midi_gen.schema.beat_state import BeatLevelState, BeatLevelSequence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# General MIDI program → role mapping
# ---------------------------------------------------------------------------
DRUM_CHANNEL = 9  # 0-indexed; channel 10 in 1-indexed MIDI

ROLE_MAP = {
    range(0, 8): "lead",        # Piano family
    range(8, 16): "lead",       # Chromatic percussion
    range(16, 24): "lead",      # Organ
    range(24, 32): "comping",   # Guitar
    range(32, 40): "bass",      # Bass
    range(40, 48): "lead",      # Strings
    range(48, 56): "lead",      # Ensemble
    range(56, 64): "lead",      # Brass
    range(64, 72): "lead",      # Reed
    range(72, 80): "lead",      # Pipe
    range(80, 88): "lead",      # Synth lead
    range(88, 96): "comping",   # Synth pad
    range(96, 104): "comping",  # Synth effects
    range(104, 112): "comping", # Ethnic
    range(112, 120): "drums",   # Percussive
    range(120, 128): "aux",     # Sound effects
}


def _program_to_role(program: int, is_percussion: bool = False) -> str:
    """Map a GM program number to a track role."""
    if is_percussion:
        return "drums"
    for rng, role in ROLE_MAP.items():
        if program in rng:
            return role
    return "aux"


# ---------------------------------------------------------------------------
# GuitarPro → intermediate note events
# ---------------------------------------------------------------------------

def _gp_track_to_notes(
    track: "guitarpro.models.Track",
    tempo: float,
) -> list[dict]:
    """Extract note events from a single GuitarPro track.

    Returns a list of dicts with keys: start, end, pitch, velocity.
    Times are in seconds.
    """
    notes: list[dict] = []
    current_time = 0.0  # in seconds
    beat_duration_s = 60.0 / tempo  # quarter-note duration

    for measure in track.measures:
        for voice in measure.voices:
            beat_offset = 0.0
            for beat in voice.beats:
                # Duration as fraction of a whole note
                dur_fraction = beat.duration.value  # e.g. 4 = quarter
                dur_seconds = (4.0 / dur_fraction) * beat_duration_s

                for note in beat.notes:
                    if note.type == guitarpro.NoteType.rest:
                        continue
                    # Compute actual pitch from string tuning + fret
                    string_tuning = track.strings[note.string - 1].value
                    pitch = string_tuning + note.value
                    if 0 <= pitch <= 127:
                        notes.append({
                            "start": current_time + beat_offset,
                            "end": current_time + beat_offset + dur_seconds,
                            "pitch": int(pitch),
                            "velocity": int(note.velocity),
                        })
                beat_offset += dur_seconds
        # Advance time by measure duration
        measure_duration = sum(
            (4.0 / b.duration.value) * beat_duration_s
            for v in measure.voices
            for b in v.beats
        ) / max(len(measure.voices), 1)
        current_time += measure_duration

    return notes


def _estimate_beats_per_bar(song: "guitarpro.models.Song") -> int:
    """Estimate the beats per bar from the first measure header."""
    if song.measureHeaders:
        ts = song.measureHeaders[0].timeSignature
        return ts.numerator
    return 4


def _estimate_beat_count(song: "guitarpro.models.Song") -> int:
    """Total number of beats in the song."""
    total = 0
    for mh in song.measureHeaders:
        total += mh.timeSignature.numerator
    return max(total, 1)


# ---------------------------------------------------------------------------
# Convert to BeatLevelSequence
# ---------------------------------------------------------------------------

def gp_file_to_sequence(
    filepath: str | Path,
    tier: str = "gold",
) -> Optional[dict]:
    """Parse a GuitarPro file and return a serialisable dict.

    The dict contains:
      - ``metadata``: file info, tempo, time signature, tier
      - ``beats``: list of BeatLevelState dicts
      - ``tracks``: per-track note events for the multitrack realizer
    """
    filepath = Path(filepath)
    try:
        song = guitarpro.parse(str(filepath))
    except Exception as exc:
        log.warning("Failed to parse %s: %s", filepath.name, exc)
        return None

    tempo = song.tempo if song.tempo else 120
    beats_per_bar = _estimate_beats_per_bar(song)
    total_beats = _estimate_beat_count(song)

    # --- Extract per-track notes and assign roles ---
    track_data: dict[str, list[dict]] = {}
    for idx, track in enumerate(song.tracks):
        if not track.measures:
            continue
        is_perc = track.isPercussionTrack
        program = track.channel.instrument if track.channel else 0
        role = _program_to_role(program, is_perc)
        label = f"{role}_{idx}"
        notes = _gp_track_to_notes(track, tempo)
        if notes:
            track_data[label] = {
                "role": role,
                "program": program,
                "is_percussion": is_perc,
                "name": track.name,
                "notes": notes,
            }

    if not track_data:
        log.warning("No usable tracks in %s", filepath.name)
        return None

    # --- Build beat-level states ---
    beat_states: list[dict] = []
    for beat_idx in range(total_beats):
        bar_idx = beat_idx // beats_per_bar
        beat_in_bar = beat_idx % beats_per_bar

        # Determine boundary level
        if beat_idx == 0:
            boundary = 3  # song start
        elif beat_in_bar == 0 and bar_idx % 8 == 0:
            boundary = 2  # section boundary (every 8 bars)
        elif beat_in_bar == 0 and bar_idx % 4 == 0:
            boundary = 1  # phrase boundary (every 4 bars)
        else:
            boundary = 0

        state = BeatLevelState(
            meter=beats_per_bar,
            beat_position=beat_in_bar,
            boundary_level=boundary,
            region_key=0,  # placeholder — needs harmonic analysis
            chord=0,       # placeholder
            harmonic_role=0,
            melodic_head=60,
            groove_token=0,
        )
        beat_states.append(asdict(state))

    # --- Assemble output ---
    result = {
        "metadata": {
            "source_file": filepath.name,
            "source_format": "guitarpro",
            "tier": tier,
            "tempo_bpm": tempo,
            "beats_per_bar": beats_per_bar,
            "total_beats": total_beats,
            "total_bars": total_beats // beats_per_bar,
            "num_tracks": len(track_data),
            "track_roles": {k: v["role"] for k, v in track_data.items()},
        },
        "beats": beat_states,
        "tracks": {
            k: {"role": v["role"], "program": v["program"],
                "name": v["name"], "note_count": len(v["notes"])}
            for k, v in track_data.items()
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GuitarPro files into the Gold corpus tier."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .gp3/.gp4/.gp5 files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write JSON output files",
    )
    parser.add_argument(
        "--tier", default="gold", choices=["gold", "silver"],
        help="Data tier label (default: gold)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively scan input directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".gp3", ".gp4", ".gp5"}
    if args.recursive:
        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]
    else:
        files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    log.info("Found %d GuitarPro files in %s", len(files), input_dir)

    success = 0
    for filepath in sorted(files):
        result = gp_file_to_sequence(filepath, tier=args.tier)
        if result is None:
            continue
        out_path = output_dir / (filepath.stem + ".json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        success += 1
        if success % 50 == 0:
            log.info("Processed %d / %d files", success, len(files))

    log.info(
        "Done. Successfully converted %d / %d files → %s",
        success, len(files), output_dir,
    )


if __name__ == "__main__":
    main()
