#!/usr/bin/env python3
"""
Command-line interface for MIDI generation.

Usage (after ``uv run``):
    # Generate progressive rock MIDI
    uv run midi-gen --genre prog_rock --num-samples 2

    # Generate jazz fusion MIDI with specific key
    uv run midi-gen --genre jazz_fusion --key 7 --major

    # Generate with specific form
    uv run midi-gen --genre prog_rock --phrase-string i4A8B8C12S8A8o4

    # Generate with audio rendering
    uv run midi-gen --genre prog_rock --render-audio
"""

import argparse

import numpy as np

from midi_gen.core.config import GenerationConfig
from midi_gen.core.pipeline import MidiGenerationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIDI Generation Module for Progressive Rock and Jazz Fusion",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="prog_rock",
        choices=["prog_rock", "jazz_fusion"],
        help="Musical genre preset (default: prog_rock)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of pieces to generate (default: 1)",
    )
    parser.add_argument(
        "--num-measures",
        type=int,
        default=None,
        help="Number of measures (default: random 32-64)",
    )
    parser.add_argument(
        "--key",
        type=int,
        default=0,
        help="Tonic pitch class 0-11 (default: 0 = C)",
    )
    parser.add_argument(
        "--major",
        action="store_true",
        default=True,
        help="Generate in major key (default)",
    )
    parser.add_argument(
        "--minor",
        action="store_true",
        help="Generate in minor key",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Tempo in BPM (default: genre-dependent)",
    )
    parser.add_argument(
        "--phrase-string",
        type=str,
        default=None,
        help="Specify form string (e.g., i4A8B8C12S8A8o4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--render-audio",
        action="store_true",
        help="Also render MIDI to audio (WAV)",
    )
    parser.add_argument(
        "--soundfont",
        type=str,
        default=None,
        help="Path to SoundFont (.sf2) file for audio rendering",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100)",
    )
    parser.add_argument(
        "--sinkhorn-iterations",
        type=int,
        default=50,
        help="Number of Sinkhorn iterations for SB solver (default: 50)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file",
    )

    args = parser.parse_args()

    # Create pipeline
    if args.config:
        pipeline = MidiGenerationPipeline.from_config_file(args.config)
    else:
        pipeline = MidiGenerationPipeline.from_preset(args.genre)

    # Apply CLI overrides
    if args.seed is not None:
        pipeline.config.seed = args.seed
        pipeline.rng = np.random.default_rng(args.seed)
    if args.quiet:
        pipeline.config.verbose = False
    pipeline.config.bridge.num_diffusion_steps = args.diffusion_steps
    pipeline.config.bridge.sinkhorn_iterations = args.sinkhorn_iterations

    is_major = not args.minor

    # Generate
    print(f"\n{'=' * 60}")
    print(f"  MIDI Generation Module v0.1.0")
    print(f"  Genre: {args.genre}")
    print(f"  Key: {args.key} ({'major' if is_major else 'minor'})")
    print(f"  Samples: {args.num_samples}")
    print(f"{'=' * 60}\n")

    results = pipeline.generate(
        num_measures=args.num_measures,
        key=args.key,
        is_major=is_major,
        phrase_string=args.phrase_string,
        bpm=args.bpm,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )

    # Optional audio rendering
    if args.render_audio:
        from midi_gen.rendering.audio_renderer import AudioRenderer

        renderer = AudioRenderer(soundfont_path=args.soundfont, sample_rate=44100)
        print(f"\n[Audio] Rendering backend: {renderer.backend}")
        for result in results:
            audio_path = renderer.render(result["midi_path"])
            result["audio_path"] = audio_path
            print(f"[Audio] Rendered: {audio_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Generation Summary")
    print(f"{'=' * 60}")
    for i, result in enumerate(results):
        print(f"\n  Sample {i + 1}:")
        print(f"    MIDI:     {result['midi_path']}")
        print(f"    Form:     {result['form_string']}")
        print(f"    Notes:    {result['num_notes']}")
        print(f"    Duration: {result['duration_seconds']:.1f}s")
        if "audio_path" in result:
            print(f"    Audio:    {result['audio_path']}")
    print()


if __name__ == "__main__":
    main()
