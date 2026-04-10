# MIDI Generation Module

**GTTM-informed Schrödinger Bridge MIDI Generation for Progressive Rock and Jazz Fusion**

A modular, research-oriented MIDI generation system that combines the **Generative Theory of Tonal Music (GTTM)** with **Schrödinger Bridge (SB)** inference to produce structurally coherent symbolic music. Designed as a drop-in upstream module for the [whole-song-gen](https://github.com/ZZWaang/whole-song-gen) cascaded diffusion pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   MIDI Generation Module                │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │  GTTM    │──▶│  SB      │──▶│  Diffusion Model   │  │
│  │  Prior   │   │  Solver  │   │  (Piano-Roll Gen)  │  │
│  └──────────┘   └──────────┘   └────────────────────┘  │
│       │                              │                  │
│       ▼                              ▼                  │
│  Phrase Structure             (2, T, 128) Piano-Roll    │
│  + Tension Curve              + Metadata + Form String  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │            WSG Adapter (optional)                │   │
│  │  Formats output for whole-song-gen pipeline      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Audio Renderer (optional)                │   │
│  │  FluidSynth / Pure-Python sine-wave fallback     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **GTTM Structural Prior** | Generates phrase structure, metrical grid, and tension curve using GTTM-informed rules for progressive rock and jazz fusion |
| 2 | **Schrödinger Bridge Solver** | Computes optimal harmonic trajectory between structural waypoints using Sinkhorn-regularized optimal transport |
| 3 | **Diffusion Model** | Generates detailed piano-roll texture conditioned on phrase structure and tension curve |
| 4 | **Structural Guidance** | Blends diffusion output with SB trajectory for harmonic coherence and diatonic filtering |
| 5 | **MIDI Output** | Converts piano-roll to MIDI with metadata, compatible with whole-song-gen format |
| 6 | **Audio Rendering** (optional) | Renders MIDI to WAV via FluidSynth or pure-Python synthesis |

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
- (Optional) PyTorch >= 2.0 for GPU-accelerated generation

---

## Quick Start

### Command Line

The package installs a `midi-gen` CLI tool that you can run via `uv`:

```bash
# Generate progressive rock MIDI (32 measures, C major)
uv run midi-gen --genre prog_rock --num-measures 32 --key 0

# Generate jazz fusion MIDI (24 measures, F major, with audio)
uv run midi-gen --genre jazz_fusion --num-measures 24 --key 5 --render-audio

# Generate with specific form structure
uv run midi-gen --genre prog_rock --phrase-string i4A8B8C12S8A8o4

# Reproducible generation with seed
uv run midi-gen --genre prog_rock --seed 42 --num-samples 3
```

### Python API

```python
from midi_gen.core.pipeline import MidiGenerationPipeline

# Create pipeline from preset
pipeline = MidiGenerationPipeline.from_preset("prog_rock")

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
    print(f"Notes: {result['num_notes']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")

    # Access whole-song-gen compatible data
    wsg_input = result['whole_song_gen_input']
    print(f"WSG phrase string: {wsg_input['phrase_string']}")
```

### Integration with whole-song-gen

```python
from midi_gen.core.wsg_adapter import WholeSongGenAdapter

# Initialize adapter with path to whole-song-gen repository
adapter = WholeSongGenAdapter("/path/to/whole-song-gen")

# Option 1: Provide form-level input
form_input = adapter.create_form_input(
    phrase_string="i4A8B8C12S8A8o4",
    key=0,
    is_major=True
)

# Option 2: Run whole-song-gen pipeline directly (requires pretrained models)
adapter.run_whole_song_gen(
    phrase_string="i4A8B8C12S8A8o4",
    key=0,
    is_major=True,
    output_dir="demo_output"
)
```

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
| Prolongational Weight | 1.5 | 1.2 |
| Tension Resolution | 32 | 16 |

### Custom Configuration

```python
from midi_gen.core.config import GenerationConfig

# Load from JSON
config = GenerationConfig.from_json("configs/prog_rock.json")

# Modify parameters
config.structure.bpm = 120.0
config.bridge.num_diffusion_steps = 200
config.tonal.default_key = 7  # G

# Create pipeline
pipeline = MidiGenerationPipeline(config)
```

See `configs/prog_rock.json` and `configs/jazz_fusion.json` for full configuration examples.

---

## Output Format

Each generation produces a directory containing:

```
output/prog_rock_20260410_s0/
├── generation.mid          # MIDI file
├── generation.wav          # Audio file (if --render-audio)
├── metadata.json           # Full generation metadata
├── form.txt                # Human-readable form description
├── piano_roll.npy          # Raw piano-roll array (2, T, 128)
└── tension_curve.npy       # Tension curve array
```

### Piano-Roll Format

The piano-roll uses the same `(2, T, 128)` format as whole-song-gen:
- **Channel 0**: Onset events (binary)
- **Channel 1**: Sustain events (binary)
- **T**: Number of time steps (16th-note resolution at 4 steps per beat)
- **128**: Full MIDI pitch range

### whole-song-gen Compatibility

The `whole_song_gen_input` dictionary in each result contains:
- `piano_roll`: The generated `(2, T, 128)` array
- `phrase_string`: Form string (e.g., `"i4A8B8C12S8A8o4"`)
- `key`: Tonic pitch class (0-11)
- `is_major`: Boolean key mode
- `phrase_structure`: List of phrase dictionaries

---

## Theoretical Background

### GTTM (Generative Theory of Tonal Music)

The module implements four components from Lerdahl and Jackendoff's GTTM:

1. **Grouping Structure**: Hierarchical segmentation of music into phrases, informed by genre-specific templates
2. **Metrical Structure**: Multi-level beat hierarchy with configurable time signatures
3. **Time-Span Reduction**: Tonal distance computation using the pitch-class circle of fifths
4. **Prolongational Reduction**: Tension-relaxation curves that guide harmonic progression

### Schrödinger Bridge

The SB solver frames music generation as an optimal transport problem:
- **Start/End States**: Defined by phrase boundary waypoints
- **Energy Function**: GTTM-based transition energy combining tonal distance, metrical congruence, tension alignment, and grouping coherence
- **Sinkhorn Regularization**: Ensures smooth, entropy-regularized trajectories
- **Coarse-to-Fine**: Operates at measure-level resolution, then upsamples

### Diffusion Model

A lightweight diffusion model generates detailed piano-roll textures:
- Linear or cosine noise schedule
- Tension-guided noise prediction
- Post-processing with onset/sustain consistency enforcement
- Diatonic filtering based on key signature

---

## Project Structure

```
MIDI-generation/
├── pyproject.toml             # uv project configuration
├── uv.lock                    # Locked dependencies
├── README.md
├── LICENSE
├── src/
│   └── midi_gen/              # Main Python package
│       ├── core/              # Pipeline and config
│       ├── gttm/              # Structural prior
│       ├── bridge/            # SB solver
│       ├── models/            # Diffusion model
│       ├── utils/             # MIDI I/O
│       └── rendering/         # Audio rendering
├── configs/                   # JSON presets
├── tests/                     # Pytest suite
├── scripts/                   # Standalone scripts
└── docs/
    └── research/              # Tracked research documents
```

---

## Development Roadmap

### Current State (v0.1.0 — Demo)

- Rule-based GTTM structural prior with genre-specific templates
- Numpy-based Schrödinger Bridge solver with Sinkhorn regularization
- Lightweight diffusion model (untrained, random initialization)
- Full pipeline with MIDI output and optional audio rendering
- whole-song-gen format compatibility

### Next Steps (v0.2.0 — Trained Model)

- [ ] Train diffusion U-Net on POP909 / Lakh MIDI datasets
- [ ] Implement PyTorch-based U-Net with attention layers
- [ ] Add GPU acceleration for SB solver
- [ ] Integrate with whole-song-gen pretrained models
- [ ] Add evaluation metrics (FID, pitch-class histogram distance)

### Future (v1.0.0 — Production)

- [ ] Fine-tune on progressive rock and jazz fusion MIDI corpora
- [ ] Implement learned GTTM prior (replace rule-based)
- [ ] Add real-time generation mode
- [ ] Support for multi-track generation (melody, bass, chords, drums)
- [ ] Integration with MIDI-to-audio neural synthesizers (DiffSynth, MIDI-DDSP)

---

## Testing

```bash
# Run all tests using uv
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src/midi_gen
```

---

## References

1. Lerdahl, F., & Jackendoff, R. (1983). *A Generative Theory of Tonal Music*. MIT Press.
2. Wang, Z., et al. (2024). *Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models*. AAAI.
3. De Bortoli, V., et al. (2021). *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*. NeurIPS.
4. Ho, J., et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
