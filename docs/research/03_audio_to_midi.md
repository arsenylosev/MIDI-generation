# Audio-to-MIDI Transcription Model Survey (2023-2025)

## Executive Summary
The landscape of Automatic Music Transcription (AMT) has undergone a fundamental transformation between late 2022 and Q1 2025. The field has shifted from discriminative deep learning frameworks (CNNs, RNNs treating transcription as frame-level classification) to **generative transcription** — reframing AMT as a sequence generation and refinement problem.

Key models covered: **Noise-to-Notes (N2N)**, **D3RM**, **MR-MT3**
Key tools: **RipX**, **Samplab**, **NeuralNote**, **Spotify's Basic Pitch**

## Key Models:
1. **Noise-to-Notes (N2N)** - Generative approach to percussion transcription, uses diffusion + MERT embeddings
2. **D3RM** - Discrete Denoising Diffusion Refinement Model for piano transcription, uses neighborhood attention
3. **MR-MT3** - Solves instrument leakage in multi-instrument transcription, memory retention mechanism
4. **Spotify's Basic Pitch** - Lightweight open-source standard
5. **NeuralNote** - Real-time implementation challenge

## Document Structure:
- Theoretical foundations (discriminative limits, diffusion mechanism, foundation models as conditioners)
- Drum transcription (N2N)
- Piano and multi-instrument systems (D3RM, MR-MT3)
- Multimodal and specialized transcription
- Open source ecosystem (Basic Pitch, NeuralNote)
- Industrial adoption (RipX DAW, Samplab, Staccato, Dubler 2)
- Technical comparison and benchmarking

Note: This document covers audio→MIDI direction (reverse of what we need for the main task, but useful for understanding the full pipeline).
