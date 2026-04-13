# MIDI Generation Module

**GTTM-informed Schrödinger Bridge MIDI Generation for Progressive Rock and Jazz Fusion**

A modular, research-oriented MIDI generation system that combines the **Generative Theory of Tonal Music (GTTM)** with **Schrödinger Bridge (SB)** inference to produce structurally coherent symbolic music. Designed as a drop-in upstream module for the [whole-song-gen](https://github.com/ZZWaang/whole-song-gen) cascaded diffusion pipeline.

---

## Architecture Overview (v0.2.0)

The v0.2.0 architecture introduces a **native beat-level state schema** and a **sparse candidate generation + learned scoring** paradigm, replacing the raw piano-roll internal representation.

```
┌─────────────────────────────────────────────────────────┐
│                   MIDI Generation Module                │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │  GTTM    │──▶│  SB      │──▶│  Sparse Candidate  │  │
│  │  Prior   │   │  Solver  │   │  Generator & Gating│  │
│  └──────────┘   └──────────┘   └────────────────────┘  │
│       │                              │                  │
│       ▼                              ▼                  │
│  Phrase Structure             Candidate Next States     │
│  + Tension Curve                     │                  │
│                                      ▼                  │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │  Corpus  │──▶│  Learned │◀──│  Compact           │  │
│  │ Ingestion│   │  Scorer  │   │  Transformer       │  │
│  └──────────┘   └──────────┘   └────────────────────┘  │
│                                      │                  │
│                                      ▼                  │
│                                Beat-Level Sequence      │
│                                      │                  │
│                                      ▼                  │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │  Guide   │◀──│Multitrack│◀──│  Texture Planner   │  │
│  │ Renderer │   │ Realizer │   │  & Note Decoder    │  │
│  └──────────┘   └──────────┘   └────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │            WSG Adapter (optional)                │   │
│  │  Formats output for whole-song-gen pipeline      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **GTTM Structural Prior** | Generates phrase structure, metrical grid, and tension curve using GTTM-informed rules for progressive rock and jazz fusion |
| 2 | **Schrödinger Bridge Solver** | Computes optimal harmonic trajectory between structural waypoints using Sinkhorn-regularized optimal transport |
| 3 | **Sparse Candidate Generator** | Proposes musically legal next beat-level states using 6 strict gating rules (meter, harmonic rhythm, region, chord, melody, groove) |
| 4 | **Learned Candidate Scorer** | Ranks candidates using a compact causal transformer trained on the corpus (with heuristic fallback) |
| 5 | **Multitrack Realizer** | Translates the beat-level sequence into full multitrack MIDI (melody, chords, bass, drums) via texture planning |
| 6 | **Guide-Audio Rendering** | Renders the multitrack MIDI into per-stem guide audio channels for downstream conditioning |
| 7 | **Evaluation Framework** | Computes structural metrics (boundary clarity, cadence arrival, harmonic rhythm) |

---

## Installation

This project uses `uv` for fast, reliable Python environment management.

```bash
# Clone the repository
git clone https://github.com/your-org/MIDI-generation.git
cd MIDI-generation

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the environment and install the package
uv sync

# Install with audio rendering support
uv sync --extra audio

# Install all optional dependencies (audio, torch, dev tools)
uv sync --all-extras
```

### Requirements

- Python >= 3.11
- NumPy >= 1.24
- pretty_midi >= 0.2.10
- (Optional) midi2audio for FluidSynth rendering
- (Optional) PyTorch >= 2.0 for the learned candidate scorer

---

## Quick Start

### Command Line

The package installs a `midi-gen` CLI tool that you can run via `uv`. The v0.2.0 pipeline is now the default.

```bash
# Generate progressive rock MIDI using the v2 pipeline
uv run midi-gen --genre prog_rock --num-measures 32 --key 0

# Generate jazz fusion MIDI with guide-audio rendering
uv run midi-gen --genre jazz_fusion --num-measures 24 --key 5 --render-audio

# Run the legacy v1 pipeline (diffusion-based)
uv run midi-gen --genre prog_rock --pipeline v1
```

### Python API (v2 Pipeline)

```python
from midi_gen.core.pipeline_v2 import PipelineV2
from midi_gen.core.config import GenerationConfig

# Load configuration
config = GenerationConfig.from_json("configs/prog_rock.json")

# Create pipeline
pipeline = PipelineV2(config)

# Generate MIDI
results = pipeline.generate(
    num_measures=32,
    key=0,           # C
    is_major=True,
    num_samples=1
)

# Access results
for result in results:
    print(f"MIDI: {result['midi_path']}")
    print(f"Form: {result['form_string']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    
    # Access structural evaluation metrics
    metrics = result['evaluation_metrics']
    print(f"Boundary Clarity: {metrics['boundary_clarity']:.2f}")
```

### Corpus Ingestion

The v0.2.0 architecture includes a corpus ingestion pipeline to train the candidate scorer:

```python
from midi_gen.corpus.ingestor import CorpusIngestor, DataTier

ingestor = CorpusIngestor()

# Ingest a directory of MIDI files into the GOLD tier
entries = ingestor.ingest_directory(
    directory="data/raw/prog_rock",
    tier=DataTier.GOLD,
    genre="prog_rock"
)

# Save the ingested corpus
ingestor.save_corpus("data/processed/corpus.json")
```

---

## Native State Schema

The core of the v0.2.0 architecture is the `BeatLevelState` schema, which represents music as a sequence of structural tokens rather than a raw piano-roll:

1. `meter_token`: Encoded time-signature class (e.g., 4/4, 7/8, 5/4)
2. `beat_position`: 0-indexed position within the current measure
3. `boundary_level`: Hierarchical boundary strength (0=none, 1=sub-phrase, 2=phrase, 3=section)
4. `region_label`: Section / key-region identifier
5. `chord_label`: Integer encoding of the current chord (root * 12 + quality)
6. `harmonic_role`: Functional role in the current key (tonic, dominant, etc.)
7. `melodic_head`: MIDI pitch of the most salient melodic note
8. `groove_token`: Categorical groove-family index

---

## Configuration

### Presets

Two genre presets are provided:

| Parameter | Progressive Rock | Jazz Fusion |
|-----------|-----------------|-------------|
| BPM | 110 | 130 |
| Default Key | C major | F major |
| Phrase Types | intro, theme_a, theme_b, development, solo, recapitulation, coda | head_in, solo_section, interlude, trading_fours, head_out, coda |
| Time Signatures | 4/4, 7/8, 5/4, 6/8 | 4/4, 7/4, 5/4 |
| Max Melody Interval | 12 (Octave) | 14 (Major 9th) |
| Chord Distance | 3 (Circle of Fifths) | 4 |

---

## Output Format

Each generation produces a directory containing:

```
output/prog_rock_20260410_s0/
├── generation.mid          # Multitrack MIDI file
├── guide_audio/            # Guide audio stems (if --render-audio)
│   ├── mix.wav
│   ├── melody.wav
│   ├── chords.wav
│   ├── bass.wav
│   └── drums.wav
├── metadata.json           # Full generation metadata & evaluation metrics
├── form.txt                # Human-readable form description
├── beat_sequence.json      # Native beat-level state sequence
└── tension_curve.npy       # Tension curve array
```

---

## Development Roadmap

### Current State (v0.2.0 — Structural Pipeline)

- Native beat-level state schema replacing raw piano-roll
- Sparse candidate generator with 6 strict gating rules
- Compact transformer candidate scorer (with heuristic fallback)
- Multitrack bar-level realizer (melody, chords, bass, drums)
- Corpus ingestion pipeline with Gold/Silver/Bronze tiers
- Guide-audio renderer for downstream conditioning
- Structural evaluation framework

### Next Steps (v0.3.0 — Training & Integration)

- [ ] Train the candidate scorer on a large corpus of progressive rock and jazz fusion MIDI
- [ ] Implement the full whole-song-gen integration using the guide-audio stems
- [ ] Add GPU acceleration for the SB solver
- [ ] Expand the multitrack realizer with more texture codes

---

## License

MIT License. See [LICENSE](LICENSE) for details.
