#!/usr/bin/env python3
"""Bronze-tier ingestion: Audio → Source Separation → Transcription → Features.

This script implements the bronze-tier data pipeline that processes audio
recordings through source separation (Demucs/HTDemucs) and automatic music
transcription (Basic Pitch) to extract structural and textural features
suitable for training the system's broad habits.

Bronze data is intentionally noisy at the note level.  Its purpose is to
teach groove families, density envelopes, rough chord movement, common
section energies, and bar-level texture types — NOT fine note-level idiom.

Usage
-----
    python scripts/bronze_pipeline.py \\
        --input-dir  data/raw/audio/ \\
        --output-dir data/bronze/ \\
        --skip-separation   # if stems already exist

Pipeline stages
---------------
1. Source separation via Demucs → drums, bass, vocals, other stems
2. Per-stem transcription via Basic Pitch → MIDI per stem
3. Beat/bar-level feature extraction → structural features JSON

Dependencies
------------
    pip install demucs basic-pitch pretty_midi librosa
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1: Source Separation (Demucs)
# ---------------------------------------------------------------------------

def run_demucs(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "cpu",
) -> Optional[Path]:
    """Run Demucs source separation on an audio file.

    Returns the path to the separated stems directory, or None on failure.
    Stems will be: drums.wav, bass.wav, vocals.wav, other.wav
    """
    try:
        cmd = [
            sys.executable, "-m", "demucs",
            "--name", model,
            "--out", str(output_dir),
            "--device", device,
            "--two-stems", "no",  # get all 4 stems
            str(audio_path),
        ]
        log.info("Running Demucs: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.warning("Demucs failed for %s: %s", audio_path.name, result.stderr[:500])
            return None

        # Demucs outputs to: output_dir / model / track_name /
        stem_dir = output_dir / model / audio_path.stem
        if stem_dir.exists():
            return stem_dir
        # Fallback: search for the stems
        for d in (output_dir / model).iterdir():
            if d.is_dir():
                return d
        return None

    except FileNotFoundError:
        log.error(
            "Demucs not found. Install with: pip install demucs"
        )
        return None
    except subprocess.TimeoutExpired:
        log.warning("Demucs timed out for %s", audio_path.name)
        return None


# ---------------------------------------------------------------------------
# Stage 2: Per-Stem Transcription (Basic Pitch)
# ---------------------------------------------------------------------------

def transcribe_stem(
    stem_path: Path,
    output_dir: Path,
    stem_name: str,
) -> Optional[Path]:
    """Transcribe a single audio stem to MIDI using Basic Pitch.

    Returns the path to the output MIDI file, or None on failure.
    """
    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH

        midi_dir = output_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)

        predict_and_save(
            audio_path_list=[str(stem_path)],
            output_directory=str(midi_dir),
            save_midi=True,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        # Basic Pitch saves as <stem_name>_basic_pitch.mid
        expected = midi_dir / f"{stem_path.stem}_basic_pitch.mid"
        if expected.exists():
            final_path = midi_dir / f"{stem_name}.mid"
            expected.rename(final_path)
            return final_path

        # Fallback: find any .mid file
        for f in midi_dir.glob("*.mid"):
            return f
        return None

    except ImportError:
        log.error("basic-pitch not found. Install with: pip install basic-pitch")
        return None
    except Exception as exc:
        log.warning("Transcription failed for %s: %s", stem_path.name, exc)
        return None


def transcribe_stem_cli(
    stem_path: Path,
    output_dir: Path,
    stem_name: str,
) -> Optional[Path]:
    """Transcribe using Basic Pitch CLI as a fallback."""
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            sys.executable, "-m", "basic_pitch",
            str(midi_dir),
            str(stem_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None

        for f in midi_dir.glob("*.mid"):
            final_path = midi_dir / f"{stem_name}.mid"
            if f != final_path:
                f.rename(final_path)
            return final_path
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 3: Feature Extraction
# ---------------------------------------------------------------------------

def extract_bar_features(
    midi_path: Path,
    stem_role: str,
    tempo_bpm: float = 120.0,
    beats_per_bar: int = 4,
) -> list[dict]:
    """Extract bar-level features from a transcribed MIDI stem.

    Features per bar:
      - note_density: number of note onsets
      - pitch_mean: average MIDI pitch
      - pitch_range: max - min pitch
      - velocity_mean: average velocity
      - rhythmic_density: fraction of 16th-note slots occupied
    """
    try:
        import pretty_midi
    except ImportError:
        return []

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []

    bar_duration = (60.0 / tempo_bpm) * beats_per_bar
    total_time = pm.get_end_time()
    num_bars = max(1, int(total_time / bar_duration) + 1)

    bars: list[dict] = []
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)

    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_duration
        bar_end = bar_start + bar_duration

        bar_notes = [
            n for n in all_notes
            if n.start >= bar_start and n.start < bar_end
        ]

        if not bar_notes:
            bars.append({
                "bar": bar_idx,
                "role": stem_role,
                "note_density": 0,
                "pitch_mean": 0,
                "pitch_range": 0,
                "velocity_mean": 0,
                "rhythmic_density": 0.0,
            })
            continue

        pitches = [n.pitch for n in bar_notes]
        velocities = [n.velocity for n in bar_notes]

        # Rhythmic density: fraction of 16th-note grid slots with onsets
        slots_per_bar = beats_per_bar * 4  # 16th notes
        slot_duration = bar_duration / slots_per_bar
        occupied_slots = set()
        for n in bar_notes:
            slot = int((n.start - bar_start) / slot_duration)
            occupied_slots.add(min(slot, slots_per_bar - 1))

        bars.append({
            "bar": bar_idx,
            "role": stem_role,
            "note_density": len(bar_notes),
            "pitch_mean": round(sum(pitches) / len(pitches), 1),
            "pitch_range": max(pitches) - min(pitches),
            "velocity_mean": round(sum(velocities) / len(velocities), 1),
            "rhythmic_density": round(len(occupied_slots) / slots_per_bar, 3),
        })

    return bars


def estimate_tempo(audio_path: Path) -> float:
    """Estimate the tempo of an audio file using librosa."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=22050, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    except Exception:
        return 120.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_audio_file(
    audio_path: Path,
    output_dir: Path,
    skip_separation: bool = False,
    device: str = "cpu",
) -> Optional[dict]:
    """Run the full bronze pipeline on a single audio file.

    Returns a summary dict, or None on failure.
    """
    track_dir = output_dir / audio_path.stem
    track_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Source Separation ---
    stem_roles = {
        "drums": "drums",
        "bass": "bass",
        "vocals": "lead",
        "other": "comping",
    }

    if skip_separation:
        # Look for pre-existing stems
        stem_dir = track_dir / "stems"
        if not stem_dir.exists():
            log.warning("No stems found for %s and --skip-separation is set", audio_path.name)
            return None
    else:
        stem_dir = track_dir / "stems"
        stem_dir.mkdir(parents=True, exist_ok=True)
        result_dir = run_demucs(audio_path, stem_dir, device=device)
        if result_dir is None:
            log.warning("Source separation failed for %s", audio_path.name)
            return None
        stem_dir = result_dir

    # --- Estimate tempo ---
    tempo = estimate_tempo(audio_path)
    log.info("Estimated tempo for %s: %.1f BPM", audio_path.name, tempo)

    # --- Stage 2: Per-Stem Transcription ---
    stem_midis: dict[str, Path] = {}
    for stem_name, role in stem_roles.items():
        if stem_name == "drums":
            continue  # Skip drum transcription (onset-based, not pitch)

        stem_path = stem_dir / f"{stem_name}.wav"
        if not stem_path.exists():
            continue

        midi_path = transcribe_stem(stem_path, track_dir, stem_name)
        if midi_path is None:
            midi_path = transcribe_stem_cli(stem_path, track_dir, stem_name)
        if midi_path is not None:
            stem_midis[stem_name] = midi_path
            log.info("Transcribed %s → %s", stem_name, midi_path.name)

    # --- Stage 3: Feature Extraction ---
    all_bar_features: dict[str, list[dict]] = {}
    for stem_name, midi_path in stem_midis.items():
        role = stem_roles[stem_name]
        features = extract_bar_features(midi_path, role, tempo)
        if features:
            all_bar_features[stem_name] = features

    # --- Compute aggregate structural features ---
    num_bars = max(
        (max(f["bar"] for f in feats) + 1 if feats else 0)
        for feats in all_bar_features.values()
    ) if all_bar_features else 0

    aggregate_density = []
    for bar_idx in range(num_bars):
        total_density = 0
        for feats in all_bar_features.values():
            bar_feats = [f for f in feats if f["bar"] == bar_idx]
            if bar_feats:
                total_density += bar_feats[0]["note_density"]
        aggregate_density.append(total_density)

    # --- Write output ---
    result = {
        "metadata": {
            "source_file": audio_path.name,
            "source_format": "audio_transcribed",
            "tier": "bronze",
            "tempo_bpm": round(tempo, 1),
            "total_bars": num_bars,
            "stems_transcribed": list(stem_midis.keys()),
        },
        "bar_features": all_bar_features,
        "aggregate": {
            "density_envelope": aggregate_density,
            "mean_density": round(
                sum(aggregate_density) / max(len(aggregate_density), 1), 2
            ),
        },
    }

    out_path = track_dir / "features.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info("Wrote features for %s (%d bars)", audio_path.name, num_bars)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bronze-tier pipeline: audio → source separation → transcription → features."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing audio files (.wav, .mp3, .flac)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write bronze-tier output",
    )
    parser.add_argument(
        "--skip-separation", action="store_true",
        help="Skip Demucs separation (use pre-existing stems)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for Demucs inference",
    )
    parser.add_argument(
        "--extensions", default=".wav,.mp3,.flac,.ogg",
        help="Comma-separated audio file extensions to process",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(args.extensions.split(","))
    audio_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in extensions
    ]

    log.info("Found %d audio files in %s", len(audio_files), input_dir)

    results = []
    for audio_path in audio_files:
        result = process_audio_file(
            audio_path, output_dir,
            skip_separation=args.skip_separation,
            device=args.device,
        )
        if result is not None:
            results.append(result["metadata"])

    # Write pipeline summary
    summary = {
        "total_files": len(audio_files),
        "successfully_processed": len(results),
        "processed_files": results,
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Bronze pipeline complete: %d / %d files processed",
        len(results), len(audio_files),
    )


if __name__ == "__main__":
    main()
