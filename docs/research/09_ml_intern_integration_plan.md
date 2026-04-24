# ML-Intern Integration Plan for MIDI Generation Autoresearch

## Executive Summary

The `ml-intern` framework by Hugging Face is an open-source, autonomous ML engineering agent designed to research literature, write code, and execute training jobs within the Hugging Face ecosystem [1]. By integrating `ml-intern` into the MIDI Generation project, we can automate the continuous discovery of new symbolic music generation techniques, the validation of datasets, and the execution of experimental training runs for the Candidate Scorer and Multitrack Realizer modules.

This document outlines the architecture of `ml-intern`, its relevant capabilities, and a concrete plan for deploying it as an autoresearch engine for our General Theory of Tonal Music (GTTM) and Schrödinger Bridge (SB) pipeline.

## Framework Capabilities and Architecture

The `ml-intern` agent operates on a queue-based asynchronous architecture powered by LiteLLM, supporting up to 300 iterations per session [2]. It is specifically engineered to overcome the limitations of standard coding assistants by leading with literature mining rather than hallucinating outdated API calls [3].

### Core Components

The system is built around several key modules that make it ideal for autoresearch:

1. **Research Sub-agent (`research_tool.py`)**: A dedicated sub-agent that operates in an isolated context window (up to 190k tokens) to prevent polluting the main agent's memory [4]. It is instructed to perform deep literature crawls starting from anchor papers, trace citation graphs downstream, read methodology sections, and extract exact training recipes and datasets [4].
2. **Paper Discovery (`papers_tool.py`)**: Integrates with Semantic Scholar and arXiv to search for papers, retrieve citation graphs, and perform semantic snippet searches across 12M+ full-text paper passages [5].
3. **Compute Integration (`jobs_tool.py`)**: Allows the agent to autonomously launch, monitor, and retrieve logs from training jobs on Hugging Face compute infrastructure, supporting hardware ranging from basic CPUs to A100 and H100 GPUs [6].
4. **Code and Doc Mining**: Tools to search GitHub for working examples (`github_find_examples`) and explore current Hugging Face documentation (`explore_hf_docs`) to ensure implementation code uses up-to-date APIs [4].

## Integration Strategy for MIDI Generation

To leverage `ml-intern` for the MIDI Generation project, we will deploy it in a headless, scheduled capacity to continuously monitor the symbolic music generation landscape and propose concrete improvements to our pipeline.

### 1. Configuration and Setup

The agent requires API keys for the LLM provider (Anthropic Claude 3.5 Sonnet is recommended for the main agent, with Haiku for the research sub-agent), Hugging Face, and GitHub [1]. 

We will configure `ml-intern` with a custom system prompt extension that provides context about our specific architecture:
> "You are researching improvements for a symbolic music generation pipeline that uses a GTTM-informed structural prior, a Schrödinger Bridge solver for harmonic trajectories, and a Candidate Scorer. Focus on papers addressing long-range coherence, multitrack arrangement, and discrete diffusion models for MIDI."

### 2. Autoresearch Workflows

We will implement three distinct autoresearch workflows using `ml-intern`'s headless mode:

#### Workflow A: Literature Mining for Candidate Scoring
**Prompt:** "Conduct a deep literature crawl starting from recent papers on 'symbolic music generation' and 'discrete diffusion for MIDI'. Find the best training recipes for scoring or ranking structural music candidates. Extract the exact datasets used, the model architectures (e.g., compact transformers), and the hyperparameters. Validate if the datasets exist on the Hugging Face Hub."

**Expected Output:** A ranked list of training recipes for the Candidate Scorer, complete with benchmark results and verified dataset links.

#### Workflow B: Dataset Discovery and Validation
**Prompt:** "Search the Hugging Face Hub and recent literature for datasets containing multitrack MIDI files, specifically focusing on progressive rock and jazz fusion genres. Use `hf_inspect_dataset` to verify the schema. We need datasets that separate drums, bass, guitars, and keys."

**Expected Output:** A curated list of Gold and Silver tier datasets that can be ingested by our `CorpusIngestor`.

#### Workflow C: Experimental Training Runs
**Prompt:** "Based on the best recipe found for transformer-based MIDI candidate scoring, write a training script using the `transformers` and `datasets` libraries. Submit this as a training job using `jobs_tool` on a T4-small GPU. Monitor the job and report the final validation loss."

**Expected Output:** Autonomous execution of training experiments, with the resulting model weights pushed to the Hugging Face Hub.

### 3. Execution and Review Cycle

To maintain control over the project direction while benefiting from automation:

1. **Scheduled Execution**: Run the literature mining workflows weekly via a cron job or GitHub Action.
2. **Human-in-the-Loop Review**: The agent will generate markdown reports (similar to our `08_data_collection_strategy.md`) summarizing its findings. The engineering team will review these reports during sprint planning.
3. **Approved Experiments**: When a proposed training recipe looks promising, the team will trigger Workflow C, allowing `ml-intern` to write the code and launch the compute job.

## Conclusion

Integrating `ml-intern` transforms our research process from a manual, time-consuming task into an automated, continuous pipeline. By leveraging its specialized research sub-agent and deep integration with the Hugging Face ecosystem, we can ensure our GTTM and Schrödinger Bridge implementation remains at the cutting edge of symbolic music generation.

## References

[1] Hugging Face. "ml-intern README." GitHub. https://raw.githubusercontent.com/huggingface/ml-intern/main/README.md
[2] Hugging Face. "ml-intern/agent README." GitHub. https://github.com/huggingface/ml-intern/tree/main/agent
[3] Hugging Face. "system_prompt_v3.yaml." GitHub. https://raw.githubusercontent.com/huggingface/ml-intern/main/agent/prompts/system_prompt_v3.yaml
[4] Hugging Face. "research_tool.py." GitHub. https://raw.githubusercontent.com/huggingface/ml-intern/main/agent/tools/research_tool.py
[5] Hugging Face. "papers_tool.py." GitHub. https://raw.githubusercontent.com/huggingface/ml-intern/main/agent/tools/papers_tool.py
[6] Hugging Face. "jobs_tool.py." GitHub. https://raw.githubusercontent.com/huggingface/ml-intern/main/agent/tools/jobs_tool.py
