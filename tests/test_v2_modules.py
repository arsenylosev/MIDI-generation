"""Tests for the revised v2 modules: schema, candidates, scorer, realizer,
corpus, guide rendering, and evaluation."""

import json
import os
import tempfile

import numpy as np
import pytest

# ── Schema ──────────────────────────────────────────────────────────────

from midi_gen.schema.beat_state import (
    BeatLevelState,
    BeatLevelSequence,
    encode_chord,
    decode_chord,
    METER_BEATS,
)
from midi_gen.schema.bar_texture import BarTextureCode, ArrangementPlan, TrackRole
from midi_gen.schema.converters import sequence_to_piano_roll, piano_roll_to_sequence


def _make_sequence(n_bars=4, bpm=120, genre="prog_rock"):
    """Helper: build a minimal BeatLevelSequence."""
    states = []
    for bar in range(n_bars):
        for beat in range(4):
            states.append(BeatLevelState(
                meter_token=0,
                beat_position=beat,
                boundary_level=2 if bar == 0 and beat == 0 else 0,
                region_label=0,
                chord_label=encode_chord(0, 0),
                harmonic_role=0,
                melodic_head=60 + beat,
                groove_token=0,
            ))
    return BeatLevelSequence(states=states, bpm=bpm, genre=genre)


class TestBeatLevelState:
    def test_to_vector_roundtrip(self):
        s = BeatLevelState(0, 2, 1, 0, encode_chord(7, 2), 2, 67, 3)
        vec = s.to_vector()
        assert vec.shape == (8,)
        assert vec[0] == 0
        assert vec[1] == 2
        assert vec[4] == encode_chord(7, 2)

    def test_encode_decode_chord(self):
        for root in range(12):
            for qual in range(9):
                code = encode_chord(root, qual)
                r, q = decode_chord(code)
                assert r == root
                assert q == qual


class TestBeatLevelSequence:
    def test_length(self):
        seq = _make_sequence(4)
        assert len(seq) == 16

    def test_to_matrix(self):
        seq = _make_sequence(4)
        mat = seq.to_matrix()
        assert mat.shape == (16, 8)

    def test_json_roundtrip(self):
        seq = _make_sequence(4, genre="jazz_fusion")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.to_json(path)
            loaded = BeatLevelSequence.from_json(path)
            assert len(loaded) == len(seq)
            assert loaded.bpm == seq.bpm
            assert loaded.genre == seq.genre
        finally:
            os.unlink(path)


class TestConverters:
    def test_sequence_to_piano_roll(self):
        seq = _make_sequence(4)
        pr = sequence_to_piano_roll(seq)
        assert pr.shape[0] == 2
        assert pr.shape[2] == 128

    def test_piano_roll_to_sequence(self):
        seq = _make_sequence(4)
        pr = sequence_to_piano_roll(seq)
        seq2 = piano_roll_to_sequence(pr, bpm=120)
        assert len(seq2) > 0


# ── Candidates ──────────────────────────────────────────────────────────

from midi_gen.candidates.generator import SparseCandidateGenerator
from midi_gen.candidates.gating import (
    MeterGate,
    HarmonicRhythmGate,
    ChordGate,
    GrooveGate,
)


class TestGating:
    def test_meter_gate(self):
        gate = MeterGate()
        state = BeatLevelState(0, 3, 0, 0, 0, 0, 60, 0)
        candidates = gate.filter(state, [
            BeatLevelState(0, 0, 0, 0, 0, 0, 60, 0),  # valid: next beat
            BeatLevelState(0, 2, 0, 0, 0, 0, 60, 0),  # invalid: skips beat
        ])
        assert len(candidates) == 1
        assert candidates[0].beat_position == 0

    def test_chord_gate(self):
        gate = ChordGate()
        state = BeatLevelState(0, 0, 0, 0, encode_chord(0, 0), 0, 60, 0)
        c_maj = BeatLevelState(0, 1, 0, 0, encode_chord(0, 0), 0, 60, 0)
        g_maj = BeatLevelState(0, 1, 0, 0, encode_chord(7, 0), 0, 60, 0)
        fsharp_dim = BeatLevelState(0, 1, 0, 0, encode_chord(6, 5), 0, 60, 0)
        results = gate.filter(state, [c_maj, g_maj, fsharp_dim])
        # At least C and G should pass (diatonic); F# dim may or may not
        assert len(results) >= 2


class TestSparseCandidateGenerator:
    def test_generate_candidates(self):
        gen = SparseCandidateGenerator(genre="prog_rock", max_candidates=32)
        current = BeatLevelState(0, 3, 0, 0, encode_chord(0, 0), 0, 60, 0)
        context_states = [
            BeatLevelState(0, b, 0, 0, encode_chord(0, 0), 0, 60, 0)
            for b in range(4)
        ]
        context = BeatLevelSequence(states=context_states, bpm=120)
        candidates = gen.generate(current, context=context)
        assert len(candidates) > 0
        assert len(candidates) <= 32


# ── Scorer ──────────────────────────────────────────────────────────────

from midi_gen.scorer.model import CandidateScorer


class TestCandidateScorer:
    def test_score_candidates(self):
        scorer = CandidateScorer(embed_dim=16, n_heads=2, n_layers=1)
        context = _make_sequence(4)
        candidates = [
            BeatLevelState(0, 0, 0, 0, encode_chord(0, 0), 0, 60, 0),
            BeatLevelState(0, 0, 0, 0, encode_chord(7, 2), 2, 67, 0),
        ]
        scores = scorer.score(context, candidates)
        assert len(scores) == 2
        assert all(isinstance(s, (float, np.floating)) for s in scores)


# ── Realizer ────────────────────────────────────────────────────────────

from midi_gen.realizer.texture_planner import TexturePlanner
from midi_gen.realizer.note_decoder import NoteDecoder
from midi_gen.realizer.realizer import MultitrackRealizer


class TestTexturePlanner:
    def test_plan_prog_rock(self):
        planner = TexturePlanner(genre="prog_rock")
        seq = _make_sequence(8)
        plan = planner.plan(seq)
        assert len(plan.bars) > 0
        assert plan.genre == "prog_rock"

    def test_plan_jazz_fusion(self):
        planner = TexturePlanner(genre="jazz_fusion")
        seq = _make_sequence(8, genre="jazz_fusion")
        plan = planner.plan(seq)
        assert len(plan.bars) > 0


class TestNoteDecoder:
    def test_decode_bar(self):
        decoder = NoteDecoder(genre="prog_rock")
        texture = BarTextureCode(bar_index=0, drums=0, bass=0, comping=0, lead=0, energy=0.5)
        bar_beats = [
            BeatLevelState(0, b, 0, 0, encode_chord(0, 0), 0, 60 + b, 0)
            for b in range(4)
        ]
        events = decoder.decode_bar(texture, bar_beats)
        assert len(events.events) > 0


class TestMultitrackRealizer:
    def test_realize(self):
        realizer = MultitrackRealizer(genre="prog_rock")
        seq = _make_sequence(8)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            pm = realizer.realize(seq, output_path=path)
            assert pm is not None
            assert os.path.exists(path)
            assert pm.get_end_time() > 0
        finally:
            os.unlink(path)


# ── Corpus ──────────────────────────────────────────────────────────────

from midi_gen.corpus.ingestor import CorpusIngestor, DataTier
from midi_gen.corpus.normalizer import EventNormalizer


class TestCorpusIngestor:
    def test_ingest_nonexistent(self):
        ingestor = CorpusIngestor()
        result = ingestor.ingest_file("/nonexistent.mid")
        assert result is None

    def test_ingest_and_save(self):
        # Create a minimal MIDI file first
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0, end=1))
        pm.instruments.append(inst)

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "test.mid")
            pm.write(midi_path)

            ingestor = CorpusIngestor()
            entry = ingestor.ingest_file(midi_path, tier=DataTier.GOLD)
            assert entry is not None
            assert entry.sequence is not None
            assert len(entry.sequence) > 0

            # Save and reload
            out_dir = os.path.join(tmpdir, "corpus")
            ingestor.save_corpus([entry], out_dir)
            loaded = ingestor.load_corpus(out_dir)
            assert len(loaded) == 1


class TestEventNormalizer:
    def test_normalize_gold(self):
        normalizer = EventNormalizer()
        seq = _make_sequence(4)
        normalized = normalizer.normalize(seq, tier=DataTier.GOLD)
        assert len(normalized) == len(seq)

    def test_normalize_bronze(self):
        normalizer = EventNormalizer()
        seq = _make_sequence(4)
        normalized = normalizer.normalize(seq, tier=DataTier.BRONZE)
        # Bronze should zero out melodic heads
        for state in normalized:
            assert state.melodic_head == -1

    def test_extract_scorer_examples(self):
        normalizer = EventNormalizer(context_length=4)
        seq = _make_sequence(8)
        examples = normalizer.extract_scorer_examples([seq])
        assert len(examples) > 0
        ctx, nxt = examples[0]
        assert ctx.shape == (4, 8)
        assert nxt.shape == (8,)


# ── Evaluation ──────────────────────────────────────────────────────────

from midi_gen.evaluation.structural_metrics import StructuralEvaluator
from midi_gen.evaluation.challenge_sets import ChallengeSetGenerator


class TestStructuralEvaluator:
    def test_evaluate(self):
        evaluator = StructuralEvaluator()
        seq = _make_sequence(16)
        report = evaluator.evaluate(seq)
        assert "phrase_boundary_clarity" in report.scores
        assert "cadence_arrival_quality" in report.scores
        assert 0 <= report.overall_score <= 1.0

    def test_evaluate_short_sequence(self):
        evaluator = StructuralEvaluator()
        seq = _make_sequence(1)
        report = evaluator.evaluate(seq)
        assert report.overall_score >= 0


class TestChallengeSetGenerator:
    def test_generate_prog_rock(self):
        gen = ChallengeSetGenerator()
        cases = gen.generate_all("prog_rock")
        assert len(cases) >= 5
        names = [c.name for c in cases]
        assert "four_bar_cadence" in names
        assert "odd_meter_7_8" in names

    def test_generate_jazz_fusion(self):
        gen = ChallengeSetGenerator()
        cases = gen.generate_all("jazz_fusion")
        names = [c.name for c in cases]
        assert "ii_V_I_resolution" in names


# ── V2 Pipeline ─────────────────────────────────────────────────────────

from midi_gen.core.pipeline_v2 import MidiGenerationPipelineV2
from midi_gen.core.config import GenerationConfig


class TestPipelineV2:
    def test_prog_rock_generation(self):
        config = GenerationConfig.prog_rock()
        config.verbose = False
        config.evaluation.enabled = True
        pipeline = MidiGenerationPipelineV2(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = pipeline.generate(
                num_measures=8,
                output_dir=tmpdir,
                num_samples=1,
            )
            assert len(results) == 1
            result = results[0]
            assert os.path.exists(result["midi_path"])
            assert result["num_beats"] > 0
            assert "evaluation" in result

    def test_jazz_fusion_generation(self):
        config = GenerationConfig.jazz_fusion()
        config.verbose = False
        pipeline = MidiGenerationPipelineV2(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = pipeline.generate(
                num_measures=8,
                output_dir=tmpdir,
                num_samples=1,
            )
            assert len(results) == 1
            assert os.path.exists(results[0]["midi_path"])

    def test_from_preset(self):
        pipeline = MidiGenerationPipelineV2.from_preset("prog_rock")
        assert pipeline.config.genre == "prog_rock"

    def test_config_new_sections(self):
        config = GenerationConfig.prog_rock()
        assert hasattr(config, "candidates")
        assert hasattr(config, "scorer")
        assert hasattr(config, "realizer")
        assert hasattr(config, "corpus")
        assert hasattr(config, "guide_rendering")
        assert hasattr(config, "evaluation")
