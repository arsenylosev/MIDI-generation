# State-of-the-Art Research Findings

## Key Models for MIDI Generation (2024-2025)

### 1. Whole-Song Hierarchical Generation (ICLR 2024)
- Cascaded diffusion, 4-level hierarchy, POP909 dataset
- Piano-roll representation, MIT license
- **Directly used by the team** via whole-song-gen repo

### 2. NotaGen (Feb 2025, IJCAI 2025)
- Decoder-only Transformer, CLaMP-DPO (RL alignment)
- Interleaved ABC/MIDI tokenization
- 110M/244M/516M params, CC-BY 4.0
- Classical sheet music focus, but architecture is adaptable

### 3. MuseCoco (Mid 2023)
- Two-stage: Text→Attributes→Music
- REMI tokenization, ~1.2B params, MIT
- Attribute-conditioned generation

### 4. Midi-LLM (Jan 2025)
- Llama-based LLM adapted for MIDI
- Extended vocabulary, 1B+ params, Apache 2.0

### 5. Text2MIDI (Late 2024)
- End-to-end text-to-MIDI translation
- FLAN-T5 encoder, ~272M params, MIT

### 6. SMDIM (2025)
- Symbolic Music Diffusion with Mamba
- State-space model integration with diffusion

## MIDI-to-Audio Tools
1. **FluidSynth** + SoundFont (.sf2) - most practical for demo
2. **midi2audio** Python package (wraps FluidSynth)
3. **py-meltysynth** - pure Python SoundFont synthesizer
4. **MIDI-VALLE** - neural zero-shot piano synthesis

## Strategy Considerations
- The team's existing pipeline (whole-song-gen) uses POP909 (pop songs)
- Target genres: progressive rock and jazz fusion
- Need to either: (a) retrain on prog rock/jazz fusion data, or (b) create a compatible MIDI generation module that can feed into the existing pipeline
- The GTTM + Schrödinger Bridge approach is the team's novel contribution
- Our module should generate MIDI that is compatible with the whole-song-gen format
