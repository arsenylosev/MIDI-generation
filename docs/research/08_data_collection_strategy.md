# Data Collection Strategy for MIDI Generation

This document outlines the comprehensive data collection strategy for the MIDI Generation Module, focusing on the progressive rock and jazz fusion genres. The strategy is structured around a three-tier system (Gold, Silver, and Bronze) to ensure that the learned components—specifically the candidate scorer and the multitrack realizer—receive the appropriate level of supervision without being overwhelmed by noisy or irrelevant data [1].

## The Three-Tier Data Architecture

The core philosophy of this data strategy is to explicitly separate training data by trust level rather than aggregating all available symbolic music into a single corpus. This distinction protects the project from treating noisy, automatically transcribed symbolic data as reliable ground truth for every task [1]. 

The candidate scorer requires a sparse set of structurally legal next-states to rank, rather than hallucinating entire compositions from scratch. Consequently, the data requirements are significantly more modest than those of an end-to-end generator. A curated collection of a few thousand symbolic files in the target stylistic neighborhood is sufficient for initial training [2].

### Gold Tier: Native Symbolic Sources

Gold data consists of native symbolic sources, including MIDI files, MusicXML, DAW exports, and carefully curated hand-authored files from the progressive rock and jazz fusion genres. This tier is essential for learning finer note-level idioms and reliable structural mappings. It serves as the foundation for training the candidate scorer and the more delicate components of the multitrack realizer [1].

Several concrete sources are recommended for the Gold tier:

| Source | Description | Utility for Project |
|--------|-------------|---------------------|
| **GuitarPro Archives** | Files in `.gp3`, `.gp4`, `.gp5`, and `.gpx` formats from sites like Ultimate Guitar. | Considered the single richest vein for this project. These files typically feature drums, bass, multiple guitar parts, and keyboards all notated separately, which maps directly onto the multitrack realizer's training needs [2]. |
| **Lakh MIDI Dataset** | A massive collection of over 170,000 MIDI files, with a subset matched to the Million Song Dataset [3]. | While messy, it can be filtered by genre tags (using the MSD metadata). It should be treated as a gold/silver mix and curated aggressively to extract relevant progressive rock and jazz fusion tracks [2]. |
| **iReal Pro Charts** | A community-driven collection of chord charts, particularly strong in jazz and fusion standards [4]. | Excellent for extracting harmonic vocabulary, cadential patterns, and form structure. While they lack full arrangements, they provide high-quality harmonic progressions [2]. |
| **RWC Music Database** | A copyright-cleared music database available to researchers, including specific Jazz and Pop sub-databases [5]. | Smaller in scale but features exceptionally well-annotated and professionally produced content, making it ideal for high-trust validation [2]. |

### Silver Tier: Aligned Symbolic and Audio Resources

Silver data consists of paired or aligned symbolic and audio resources where the symbolic layer is reasonably reliable and the instrumentation information is useful. This tier is particularly valuable for learning multitrack texture statistics—such as how different instrument roles interact and what density envelopes look like across different musical sections. It is also crucial for subsequent audio rendering experiments [1].

The **Synthesized Lakh (Slakh) Dataset** is a prime example of Silver tier data. Slakh2100 contains 2,100 automatically mixed tracks and accompanying aligned MIDI files, synthesized from the Lakh MIDI Dataset using professional-grade sample libraries [6]. This provides perfect alignment between high-quality audio stems and their corresponding symbolic representations, which is invaluable for training the guide-audio renderer and understanding texture density.

### Bronze Tier: Audio-Transcribed Material

Bronze data consists of audio-transcribed material obtained by running source separation and automatic transcription on recorded music. The primary goal of this tier is not to achieve legalistic, note-perfect transcription of every ornament. Instead, Bronze data is used to teach the system broad structural and textural habits, such as groove families, density envelopes, rough chord movement, common section energies, and bar-level texture types [1].

The ingestion pipeline for Bronze data requires a pragmatic approach, recognizing that different instrument roles necessitate different transcription strategies [1]. The recommended pipeline utilizes the following state-of-the-art models:

| Tool | Function | Application in Pipeline |
|------|----------|-------------------------|
| **Demucs / HTDemucs** | Source Separation | Separates the original audio mix into distinct stems: drums, bass, vocals/lead, and harmonic residue (accompaniment) [7]. |
| **Basic Pitch** | Monophonic/Sparse Transcription | A lightweight neural network by Spotify used to transcribe the isolated bass and lead-like stems into MIDI [8]. |
| **MT3** | Polyphonic Transcription | Google Magenta's Multi-Task Multitrack Music Transcription model, used for estimating harmonic content and chord labels from the accompaniment stem [9]. |

The critical design principle for the Bronze pipeline is that it must feed the project's native representation—beat-level structural states and bar-level texture codes—rather than forcing the project to inherit the errors of the transcription tools. The objective is to extract structural and textural features that the planner and realizer consume, rather than attempting to reconstruct a perfect MIDI file from audio [1].

## Implementation Strategy

To operationalize this data strategy, the repository requires a suite of ingestion scripts tailored to these specific sources. The immediate implementation priorities include:

1. A parser for GuitarPro files (`.gp5`, `.gpx`) utilizing the `pyguitarpro` library to extract multitrack arrangements into the native `BeatLevelSequence` schema.
2. A filtering utility for the Lakh MIDI Dataset that leverages Million Song Dataset metadata to isolate progressive rock and jazz fusion tracks.
3. A Bronze tier pipeline script that orchestrates Demucs for source separation and Basic Pitch for stem transcription, outputting structural features rather than raw MIDI.

These ingestion utilities will populate the Gold, Silver, and Bronze tiers, providing the necessary training data for the candidate scorer and multitrack realizer.

## References

[1] Revised Strategy Note for MIDI Corpus Modeling and MIDI-to-Audio Rendering. Internal Project Document, April 2026.
[2] User Communication regarding Data Collection Strategy. Internal Project Correspondence.
[3] Raffel, C. "The Lakh MIDI Dataset v0.1." colinraffel.com.
[4] iReal Pro Forums. "Fusion and Smooth Jazz." forums.irealpro.com.
[5] Goto, M. "RWC Music Database." staff.aist.go.jp.
[6] Manilow, E., et al. "Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity." MERL, 2019.
[7] Facebook Research. "Demucs: Music Source Separation." github.com/facebookresearch/demucs.
[8] Spotify Audio Intelligence Lab. "Basic Pitch: An open source MIDI converter." basicpitch.spotify.com.
[9] Google Magenta. "MT3: Multi-Task Multitrack Music Transcription." github.com/magenta/mt3.
