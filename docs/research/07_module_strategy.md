# Module Development Strategy: ML Music Generation for Progressive Rock and Jazz Fusion

## 1. Executive Summary
This document outlines a comprehensive development strategy for a MIDI generation module tailored to progressive rock and jazz fusion. The module is designed to integrate seamlessly with the existing `whole-song-gen` codebase, which utilizes a cascaded diffusion architecture for hierarchical music generation [1]. By incorporating a Generative Theory of Tonal Music (GTTM) structural prior and a Schrödinger Bridge (SB) inference mechanism, the proposed module addresses the challenge of generating long-form, structurally coherent, and stylistically authentic music [2].

## 2. Architectural Integration Strategy
The existing `whole-song-gen` repository implements a four-level cascaded diffusion model: Form, Counterpoint, Lead Sheet, and Accompaniment [1]. The proposed strategy introduces a novel "Initialization and Planning" module that sits upstream of this cascade, effectively replacing or augmenting the "Form" generation level with a more robust, theory-informed approach.

### 2.1 The GTTM-Informed Prior
Progressive rock and jazz fusion are characterized by complex time signatures, extended harmonic structures, and long-range thematic development. Purely data-driven models often fail to capture these nuances, resulting in derivative or structurally wandering outputs [3]. To mitigate this, the strategy employs a hybrid prior:
- **Corpus Prior:** A neural network or N-gram model trained on a curated dataset of progressive rock and jazz fusion MIDI files. This captures the idiomatic surface-level statistics of the genres.
- **Structural Prior (GTTM):** A set of computable energy features derived from Lerdahl and Jackendoff's Generative Theory of Tonal Music. This enforces hierarchical grouping, metrical structure, and prolongational tension/relaxation arcs [2].

These priors are combined to form a custom Transition Kernel ($P_0$), which guides the generation process.

### 2.2 Schrödinger Bridge Inference
To ensure long-range coherence, the module utilizes Schrödinger Bridge (SB) inference. Instead of autoregressively generating tokens, the system samples or optimizes structurally valid start and end passages (endpoints) [2]. The SB solver then computes the optimal "geodesic" path between these endpoints under the combined $P_0$ prior. This approach guarantees that the generated music reaches its intended structural goals while maintaining stylistic fidelity along the way.

## 3. Implementation Roadmap

### Phase 1: Data Infrastructure and Tokenization
The foundation of the module relies on a robust data representation capable of handling the complexities of the target genres.
- **Dataset Curation:** Assemble a dataset of progressive rock and jazz fusion MIDI files. Existing datasets like POP909 are insufficient for these genres [1].
- **Tokenization:** Implement a 12-EDO (Equal Division of the Octave) symbolic representation, with extensibility to 19-EDO for microtonal applications [2]. The tokenization scheme must capture pitch, duration, velocity, and structural markers.

### Phase 2: GTTM Feature Engineering
Translate the theoretical concepts of GTTM into computable algorithms.
- **Metrical and Grouping Analysis:** Develop algorithms to extract hierarchical beat structures and phrase boundaries from the corpus.
- **Tonal Distance Metric:** Implement a metric to quantify harmonic tension and relaxation, crucial for the affective trajectories of progressive rock.

### Phase 3: Schrödinger Bridge Solver
Develop the core inference engine.
- **Sparse Graph Construction:** Implement musically constrained candidate generation to build a time-unrolled graph of possible structural states.
- **Pruning and Gating:** Apply hard gating rules to prevent combinatorial explosion during graph construction.
- **SB Optimization:** Implement the forward-backward scaling equations to find the optimal path between endpoints.

### Phase 4: Integration and Rendering
Connect the new module to the existing `whole-song-gen` pipeline.
- **Format Conversion:** Ensure the output of the SB solver (the structural blueprint) is formatted correctly as input for the Counterpoint and Lead Sheet levels of the cascaded diffusion model.
- **MIDI-to-Audio Rendering (Optional):** Integrate a lightweight rendering solution, such as FluidSynth with a high-quality SoundFont, to provide immediate acoustic feedback of the generated MIDI [4].

## 4. Conclusion
By combining the structural rigor of GTTM with the global planning capabilities of Schrödinger Bridge inference, this strategy provides a robust framework for generating complex, long-form music in the progressive rock and jazz fusion genres. The modular design ensures seamless integration with the existing cascaded diffusion architecture, paving the way for a new generation of AI-assisted composition tools.

## References
[1] Z. Wang, L. Min, and G. Xia, "Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models," ICLR 2024. Available: https://github.com/ZZWaang/whole-song-gen
[2] "AI Music Generation Dev Plan," Internal Google Document.
[3] "Symbolic Music Generation Model Survey," Internal Google Document.
[4] "MIDI-to-Audio Synthesis Model Survey," Internal Google Document.
