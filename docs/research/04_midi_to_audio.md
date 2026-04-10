# MIDI-to-Audio Synthesis Model Survey (2023-2025)
## "Converging Paradigms in Neural Audio Synthesis: From Symbolic Conditioning to Discrete Timbre Tokenization"

## Key Approaches:
1. **Neural Codec Language Modeling** - treats audio synthesis as token-based translation (like LLMs)
   - MIDI-VALLE: Zero-shot piano synthesis via codec modeling
   - TokenSynth: Contrastive embeddings for instrument cloning
   - MusicGen and JASCO: Industrial standard

2. **Generative Refinement** - hybridizes concatenative sampling with diffusion-based denoising
   - CoSaRef: Concatenative Sampler and Refinement
   - MIAO: Multi-modal Contrastive Synthesis

3. **Industrial "Gray-Box" Solutions** - DDSP + neural filtering
   - Synplant 2, Combobulator, Mawf, Neutone

4. **Large Scale Generative Models** - prompt-based
   - Mustango: Music theory-guided diffusion
   - Stable Audio Open

## Key for our use case:
- MIDI-VALLE for piano synthesis
- MusicGen/JASCO for industrial-grade synthesis
- Mustango for music-theory-guided approach
- FluidSynth / SoundFont for simple MIDI-to-audio rendering
