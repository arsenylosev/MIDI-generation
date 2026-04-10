# AI Music: Generating Interesting AI - Executive Summary

## Executive Summary

A persistent challenge in algorithmic music composition is to generate music that (a) exhibits multi minute structure (phrases, sections, returns, development), (b) communicates affective trajectories (tension, release, anticipation, closure), and (c) remains novel rather than a close imitation but not an exact copy of the training corpus. Purely data-driven models can learn surface style effectively but may struggle with controllable long-range form or with guaranteeing structural coherence. Conversely, purely rule-driven systems can enforce coherence but risk sounding mechanical or stylistically narrow.

This project attempts to apply the following hybrid strategy to generate a better sounding AI music:

- Use a **structural prior** inspired by Generative Theory of Tonal Music (GTTM): hierarchical grouping, meter, time-space "importance," and prolongational (tension/relaxation) relations.
- Use a **corpus prior** that captures idiomatic distributions and long-tail stylistic details.
- Use a **Schrodinger bridge** to connect endpoints (start/end, or start/middle/start) by sampling or optimizing over "geodesic" musical paths under the combined prior.

Overall, the approach aims to produce music that is simultaneously coherent, expressive, and novel: coherent via structural priors and SB bridging, expressive via tension-aware rendering and audio synthesis, and novel via stochastic path distributions that are guided but not confined to corpus mimicry.

## Challenge: Structure and Meaning without Derivative Imitation

Let D be a corpus and let p_data be a learned distribution over symbolic music. A common failure mode is that high-likelihood samples under p_data are overly conservative: they remain close to frequent patterns in D and may not produce strong long-range arcs or novel large-scale designs. Meanwhile, explicit structure constraints can enforce coherence but may not capture the nuanced, local idioms that make progressive rock and jazz fusion feel authentic.

The proposed approach explicitly separates (i) what is structurally plausible (GTTM-like constraints and preferences) from (ii) what is stylistically typical (corpus statistics), and then uses SB to obtain coherent long trajectories that satisfy endpoint constraints while remaining close, in KL divergence, to a reference "prior path" distribution shaped by both influences.

## Generative Theory of Tonal Music (GTTM)

GTTM is a system of music analysis that aims to formalize aspects of what listeners infer when hearing music: hierarchical segmentation (grouping), hierarchical beat structure (meter), and reductions that assign hierarchical importance and tension relations:

1. Grouping structure: nested segmentation into motives, phrases, sections.
2. Metrical structure: multi-level strong/weak beat hierarchy.
3. Time-span reduction: given grouping and meter, choose heads (structurally important events) within spans, producing a headed tree.
4. Prolongational reduction: describe tension/relaxation relations and harmonic dependency, roughly analogous to (but distinct from) Schenkerian prolongation (the idea that a single harmonic or melodic entity can be "stretched out" over time by embedding it inside surface-level motion, without actually leaving it at a deeper structural level).

## Schrodinger Bridge (SB) Idea for Music Generation

Consider a discrete-time stochastic process (S_t)_{t=0}^T on a state space S with a reference path measure:

P_0(S_{0:T}) = μ_0(S_0) ∏_{t=0}^{T-1} K_t(S_t, S_{t+1}),

where K_t is a transition kernel. The Schrodinger bridge problem seeks a new path measure Q that matches prescribed endpoint distributions π_a and π_T while remaining as close as possible to P_0 in KL divergence:

Q* = arg min_{Q: Q_0=π_0, Q_T=π_T} KL(Q||P_0).

## Architecture Diagram (from page 3 of 5)

The document contains a detailed architecture flowchart with these main stages:

### 1. Initialization & Priors
- Configuration & Vocabularies
- Corpus Prior (Neural Net / N-Gram) → Provides p_data
- GTTM Prior (Structural Rules) → Provides f_GTTM
- Combined Prior P0 Transition Kernel

### 2. Endpoint Planning (e.g., Intro → Outro)
- Generate Start Passage (π_0)
- Generate End Passage (π_T)
- Scene Constraints

### 3. Schrödinger Bridge Inference
- Build Sparse Graph
- Apply Hard Gating
- Pruned Candidate Layers
- SB Solver
- Sample or MAP
- BeatState Trajectory S_{0:T}

### 4. Decoder (Track Pattern)
- Decoder Engine
- Extract Groove Tokens → Calculate Tension
- Sub-Beat Rhythm Grid
- Outputs: Drums, Bass Line, Comping/Chords, Lead Melody, Tension & Expression
- Apply Velocity/Microtiming
- Multi-Track Score

### 5. MIDI Rendering
- MIDI Transformer
- 12-EDO Direct → Standard MIDI File
- 19-EDO MPE / MTS → Microtonal MIDI File

## How to Read This Diagram

1. **Initialization & Priors (Top):** The system starts by combining data-driven 'habits' from a training corpus (e.g., via a Neural Net or N-Gram model) with the cognitive structural 'rules' of GTTM (E_..n) to create the custom Transition Kernel (P0).
2. **Endpoint Planning:** The system uses P0 to anchor the composition by sampling or optimizing structurally valid start and end passages (or start and middle passages for cyclic forms).
3. **Schrödinger Bridge Inference (Middle):** The system generates possible paths, aggressively prunes them using rule-based 'Hard Gating' to prevent combinatorial explosion, scores the remaining candidate transitions with P0, and uses the SB Solver to lock in the final, optimal BeatState trajectory.
4. **Decoder (Bottom):** The abstract beat-level path is broken down into a sub-beat rhythmic grid using groove-specific onset masks and accent curves. Four distinct functional generators create the specific instrument tracks. A tension scalar adds human-like expression.
5. **MIDI Rendering:** The raw score is finally translated into either standard 12-EDO MIDI or expressive, microtonal 19-EDO MIDI.

## Scope and Objectives

### Primary Objectives
- Generate 5-15 minute music pieces with multi-level structure (meter, grouping, harmonic motion, tension arcs) that remain novel, not merely derivative of training corpus.
- Support two generation plans:
  - **Method A:** Generate start and end passages, then compute SB "geodesic" between them under a combined prior.
  - **Method B:** Generate start and middle passages, compute SB from start to middle, then a second SB from middle back to start (start is also end).
- Support both:
  - Algorithmic mode: no neural model (GTTM + SB only, optionally with non-neural corpus statistics such as n-grams).
  - Hybrid mode: include an optional neural predictive model trained on symbolic music as a corpus prior.
- Support Equal Distribution of Octaves (EDO) as a simple parameter N (not hard-coded to 12), with 12 and 19 as first-class use cases.

### Non-goals (for the initial implementation)
- Direct audio generation. The system outputs symbolic scores and MIDI, which can be rendered to audio by a separate audio synthesizer.
- Full timbral modeling. Only MIDI-expressible aspects are modeled initially (velocity, articulation proxies, microtiming, per-note expressive controls if desired)

### References
1. Software Design Specification: GTTM and Schrodinger Bridge (Link)
2. GTTM-Informed Schrodinger Bridge Music Generation (Link)

Note: Document also has a "Quarterly Plan" tab.

## Quarterly Plan

The goal of this quarter is to develop a modular, 12 (Equal Division of the Octave) EDO symbolic music generation system that models musical structure as a discrete stochastic process and plans coherent long-range trajectories using Schrödinger Bridge inference. The work progresses from defining core symbolic representations and GTTM-inspired structural energies, through constrained candidate generation and sparse graph construction, to global path inference and final decoding into MIDI. The end result is an end-to-end prototype capable of generating short, structurally coherent symbolic musical passages from explicit configurations.

### Objectives and Key Results

**Objective 1: Define configs and token vocabularies (12 EDO).**
- KR1: A 12 Equal Division of the Octave(EDO) representation parameter scheme and validation of EDO pitch-class rendering correctness unit tests.
- KR2: GTTM features energies and tonal distance metric.
- KR3: Corpus prior built using N-Gram approach.

**Objective 2: Implement GTTM feature energies and tonal distance metric**
- KR1: A formalized set of GTTM preference rules and their implementation in computable features, summed into an energy for progressive rock/fusion jazz genres.

**Objective 3: Implement Candidate generation and sparse graph builder with pruning.**
- KR1: Study and document outcomes on available candidate generation strategies.
- KR2: An implementation of an edge scoring algorithm.
- KR3: An implementation of layer pruning algorithm that will be used to deduplicate identical states.
- KR4: Unit tests for candidate gating constraints.

**Objective 4: Implement Schrodinger Bridge solver algorithm and Sampling and MAP algorithms.**
- KR1: An initial implementation of the SB solver algorithm.
- KR2: A tested implementation of Sample and MAP algorithms.
- KR3: A tested working version of SB solver that converges on small graphs.

**Objective 5: Implement Decoder and MIDI rendering pipeline**
- KR1: An implementation of a symbolic score decoder that converts music score representation to MIDI note numbers.
- KR2: An implementation of a MIDI rendering algorithm that will be used to show MIDI structures.

**Objective 6: Implement Method A from the above mentioned generation plans**
- KR1: A final algorithm that, speaking at a higher level, brings together:
  - Building configs, vocabularies, tonal systems and priors.
  - Endpoint distribution generator.
  - Sparse layered graph builder.
  - SB solver.
  - Decoder to convert symbolic score so that the MIDI renderer shows the output in MIDI.

### Phase 1: Defining configs and token libraries and implementing GTTM algorithm
**Description:** This phase establishes the core symbolic and structural foundations of the system. It defines EDO-generic pitch representations, Corpus priors and a minimal but expressive set of GTTM-inspired preference rules encoded as computable energy features.

**Milestones:**
- Implement a 12 Equal Division of the Octave (EDO) as an explicit representation parameter.
- Validate EDO pitch-class rendering correctness by writing unit tests.
- Prepare GTTM preference rules implemented as computable features, summed into an energy function and a tonal distance metric.
- Prepare a corpus prior for prog/fusion genres using the N-Gram approach.
**Deadline: End of the second week (1-2)**

### Phase 2: Implementing Candidate generation and sparse graph builder
**Description:** This phase focuses on controlling combinatorial complexity by implementing musically constrained candidate generation and sparse graph construction. It defines how plausible next structural states are proposed, scored, deduplicated, and pruned at each beat. The goal is to produce a bounded, inspectable time-unrolled graph suitable for global inference without exponential growth.
**Deadline: End of the fourth week (3-4)**

### Phase 3: Implementing Schrodinger Bridge solver, Sample and MAP algorithms
**Description:** This phase implements global trajectory inference over the sparse structural graph using a Schrödinger Bridge formulation. It focuses on correctly solving the forward-backward scaling equations, deriving bridge-modified transitions, and supporting both stochastic sampling and MAP path extraction.
**Deadline: End of the ninth week (5-9)**

### Phase 4: Implementing the Method A generation plan
**Description:** This phase integrates all previously implemented components into a complete end-to-end generation pipeline. It introduces the Method A planning strategy, decodes structural trajectories into symbolic scores, and renders them as MIDI output. The result is a functional prototype capable of generating short, coherent symbolic musical passages from configuration through audible output.

**Phase 4 Milestones:**
- Prepare a MIDI rendering algorithm to be used to show MIDI structures.
- Implement the final algorithm that, speaking at a higher level, brings together:
  - Building configs, vocabularies, tonal systems and priors.
  - Endpoint distribution generator.
  - Sparse layered graph builder.
  - SB solver.
  - Decoder to convert symbolic score so that the MIDI renderer shows the output in MIDI.
**Deadline: End of the last week. Week (10-12)**

### Skill Development
**Technical Skills:** Complex tree data structures, graph operations, probabilistic modeling and ML, GTTM, modern MIDI specifications (MPE, MTS), functional programming.
**Soft Skills:** Team Collaboration, Problem Solving, Communication.
