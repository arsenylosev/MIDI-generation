"""
Tests for the MIDI generation pipeline.

Run with:
    uv run pytest tests/ -v
"""

import json
import os
import tempfile

import numpy as np

from midi_gen.bridge.schrodinger_bridge import (
    CandidateGenerator,
    MusicalState,
    SchrodingerBridgeSolver,
)
from midi_gen.core.config import GenerationConfig, StructureConfig, TonalConfig
from midi_gen.core.pipeline import MidiGenerationPipeline
from midi_gen.gttm.structural_prior import (
    BeatState,
    GroupingAnalyzer,
    GTTMPrior,
    MetricalGrid,
    TensionCurveGenerator,
    TonalDistanceMetric,
)
from midi_gen.models.diffusion_model import (
    DiffusionConfig,
    MidiDiffusionModel,
    NoiseSchedule,
)
from midi_gen.utils.midi_utils import (
    generate_form_string,
    piano_roll_to_note_list,
    trajectory_to_piano_roll,
)


def test_config_presets():
    """Test configuration presets."""
    prog = GenerationConfig.prog_rock()
    assert prog.genre == "prog_rock"
    assert prog.structure.bpm == 110.0

    jazz = GenerationConfig.jazz_fusion()
    assert jazz.genre == "jazz_fusion"
    assert jazz.structure.bpm == 130.0


def test_config_serialization():
    """Test config save/load."""
    config = GenerationConfig.prog_rock()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        config.to_json(f.name)
        loaded = GenerationConfig.from_json(f.name)
        assert loaded.genre == "prog_rock"
        os.unlink(f.name)


def test_tonal_distance():
    """Test tonal distance computation."""
    metric = TonalDistanceMetric(edo=12)

    # Same chord should have zero distance
    pc = np.zeros(12)
    pc[0] = pc[4] = pc[7] = 1  # C major
    dist = metric.chord_distance(pc, 0, pc, 0)
    assert dist == 0.0

    # C to G should be close (fifth)
    pc_g = np.zeros(12)
    pc_g[7] = pc_g[11] = pc_g[2] = 1  # G major
    dist_cg = metric.chord_distance(pc, 0, pc_g, 7)
    assert dist_cg > 0

    # C to F# should be far (tritone)
    pc_fs = np.zeros(12)
    pc_fs[6] = pc_fs[10] = pc_fs[1] = 1  # F# major
    dist_cfs = metric.chord_distance(pc, 0, pc_fs, 6)
    assert dist_cfs > dist_cg


def test_metrical_grid():
    """Test metrical grid construction."""
    grid = MetricalGrid(beats_per_measure=4, steps_per_beat=4, num_measures=4)
    assert grid.total_steps == 64

    # Downbeat should have highest weight
    assert grid.get_weight(0) == 1.0

    # Off-beat should have lower weight
    assert grid.get_weight(1) < grid.get_weight(0)


def test_grouping_analyzer():
    """Test phrase structure generation."""
    analyzer = GroupingAnalyzer(genre="prog_rock")
    rng = np.random.default_rng(42)
    phrases = analyzer.generate_phrase_structure(48, rng)

    assert len(phrases) > 0
    total = sum(p["lgth"] for p in phrases)
    assert total == 48

    # Jazz fusion
    analyzer_jazz = GroupingAnalyzer(genre="jazz_fusion")
    phrases_jazz = analyzer_jazz.generate_phrase_structure(32, rng)
    assert len(phrases_jazz) > 0


def test_tension_curve():
    """Test tension curve generation."""
    gen = TensionCurveGenerator(resolution=16)
    phrases = [
        {"name": "intro", "lgth": 4},
        {"name": "theme_a", "lgth": 8},
        {"name": "coda", "lgth": 4},
    ]
    curve = gen.generate_curve(phrases, beats_per_measure=4, steps_per_beat=4)

    assert len(curve) == 16 * 16  # 16 measures * 16 steps
    assert curve.min() >= 0.0
    assert curve.max() <= 1.0


def test_noise_schedule():
    """Test diffusion noise schedule."""
    config = DiffusionConfig(num_steps=100)
    schedule = NoiseSchedule(config)

    assert len(schedule.betas) == 100
    assert schedule.alpha_bars[0] > schedule.alpha_bars[-1]

    # Test noise addition and removal
    x0 = np.random.randn(2, 16, 128)
    noisy, noise = schedule.add_noise(x0, 50)
    assert noisy.shape == x0.shape


def test_candidate_generator():
    """Test musical candidate generation."""
    gen = CandidateGenerator(genre="prog_rock", num_candidates=16)
    rng = np.random.default_rng(42)

    state = MusicalState(
        time_step=0,
        pitch_vector=np.zeros(128),
        chord_root=0,
        chord_quality=0,
        tension=0.5,
        velocity=0.7,
        metrical_weight=1.0,
    )

    candidates = gen.generate_candidates(state, target_tension=0.6, rng=rng)
    assert len(candidates) <= 16
    assert all(isinstance(c, MusicalState) for c in candidates)


def test_diffusion_model():
    """Test diffusion model generation."""
    config = DiffusionConfig(num_steps=10, hidden_dim=64, num_layers=2)
    model = MidiDiffusionModel(config)

    rng = np.random.default_rng(42)
    piano_roll = model.generate(num_steps=32, rng=rng)

    assert piano_roll.shape == (2, 32, 128)
    assert piano_roll.min() >= 0.0
    assert piano_roll.max() <= 1.0


def test_piano_roll_conversion():
    """Test piano-roll to note list conversion."""
    # Create a simple piano-roll
    pr = np.zeros((2, 32, 128))
    pr[0, 0, 60] = 1.0  # C4 onset at step 0
    pr[1, 1, 60] = 1.0  # C4 sustain at step 1
    pr[1, 2, 60] = 1.0  # C4 sustain at step 2
    pr[0, 4, 64] = 1.0  # E4 onset at step 4

    notes = piano_roll_to_note_list(pr, bpm=120.0, steps_per_beat=4)
    assert len(notes) == 2

    # C4 should be longer than E4
    c4_note = [n for n in notes if n["pitch"] == 60][0]
    e4_note = [n for n in notes if n["pitch"] == 64][0]
    assert c4_note["end_time"] > c4_note["start_time"]


def test_form_string_generation():
    """Test form string generation."""
    phrases = [
        {"name": "intro", "type": "i", "lgth": 4},
        {"name": "theme_a", "type": "A", "lgth": 8},
        {"name": "theme_b", "type": "B", "lgth": 8},
        {"name": "coda", "type": "o", "lgth": 4},
    ]
    form = generate_form_string(phrases)
    assert form == "i4A8B8o4"


def test_full_pipeline():
    """Test the complete generation pipeline."""
    config = GenerationConfig.prog_rock()
    config.bridge.num_diffusion_steps = 10
    config.bridge.sinkhorn_iterations = 5
    config.bridge.num_candidates_per_step = 8
    config.verbose = False
    config.seed = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir
        pipeline = MidiGenerationPipeline(config)

        results = pipeline.generate(
            num_measures=16,
            key=0,
            is_major=True,
            num_samples=1,
        )

        assert len(results) == 1
        result = results[0]
        assert os.path.exists(result["midi_path"])
        assert result["num_notes"] >= 0
        assert "form_string" in result
        assert "whole_song_gen_input" in result

        # Check whole-song-gen compatibility
        wsg_input = result["whole_song_gen_input"]
        assert "phrase_string" in wsg_input
        assert "key" in wsg_input
        assert "piano_roll" in wsg_input


def test_jazz_fusion_pipeline():
    """Test jazz fusion generation."""
    config = GenerationConfig.jazz_fusion()
    config.bridge.num_diffusion_steps = 10
    config.bridge.sinkhorn_iterations = 5
    config.bridge.num_candidates_per_step = 8
    config.verbose = False
    config.seed = 123

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir
        pipeline = MidiGenerationPipeline(config)

        results = pipeline.generate(
            num_measures=16,
            key=5,
            is_major=True,
            num_samples=1,
        )

        assert len(results) == 1
        assert os.path.exists(results[0]["midi_path"])
