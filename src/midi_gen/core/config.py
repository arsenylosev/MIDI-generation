"""
Configuration module for the MIDI generation pipeline.

Defines all configurable parameters for the generation process,
including musical parameters, model hyperparameters, and output settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os


@dataclass
class TonalConfig:
    """Configuration for tonal system parameters."""
    edo: int = 12
    pitch_range: Tuple[int, int] = (21, 108)
    default_key: int = 0
    is_major: bool = True
    allowed_keys: Optional[List[int]] = None

    def pitch_classes(self) -> int:
        return self.edo

    def validate(self):
        assert self.edo in (12, 19), f"EDO must be 12 or 19, got {self.edo}"
        assert 0 <= self.default_key < self.edo


@dataclass
class StructureConfig:
    """Configuration for musical structure parameters."""
    num_beats_per_measure: int = 4
    num_steps_per_beat: int = 4
    min_measures: int = 16
    max_measures: int = 128
    bpm: float = 120.0
    phrase_types: List[str] = field(
        default_factory=lambda: ["intro", "verse", "chorus", "bridge", "solo", "outro"]
    )
    time_signatures: List[Tuple[int, int]] = field(
        default_factory=lambda: [(4, 4), (7, 8), (5, 4), (6, 8), (3, 4)]
    )


@dataclass
class GTTMConfig:
    """Configuration for GTTM-based structural prior."""
    grouping_weight: float = 1.0
    metrical_weight: float = 1.0
    time_span_weight: float = 1.0
    prolongational_weight: float = 1.0
    tension_curve_resolution: int = 16
    max_prolongational_depth: int = 6


@dataclass
class BridgeConfig:
    """Configuration for Schrödinger Bridge inference."""
    num_diffusion_steps: int = 100
    num_candidates_per_step: int = 32
    pruning_threshold: float = 0.1
    max_graph_nodes: int = 10000
    sinkhorn_iterations: int = 50
    sinkhorn_epsilon: float = 0.01
    use_map: bool = False


@dataclass
class ModelConfig:
    """Configuration for the neural network models."""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    latent_dim: int = 64
    use_pretrained_encoder: bool = False
    pretrained_encoder_path: Optional[str] = None


@dataclass
class RenderingConfig:
    """Configuration for MIDI and audio rendering."""
    soundfont_path: Optional[str] = None
    sample_rate: int = 44100
    output_format: str = "wav"
    velocity_default: int = 100
    use_fluidsynth: bool = True


@dataclass
class CandidateConfig:
    """Configuration for the sparse candidate generator."""
    max_candidates: int = 64
    meter_gating: bool = True
    harmonic_rhythm_gating: bool = True
    region_gating: bool = True
    chord_gating: bool = True
    melody_gating: bool = True
    groove_gating: bool = True


@dataclass
class ScorerConfig:
    """Configuration for the learned candidate scorer."""
    embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 3
    context_length: int = 16
    checkpoint_path: Optional[str] = None
    use_scorer: bool = False  # disabled until trained


@dataclass
class RealizerConfig:
    """Configuration for the multitrack bar-level realizer."""
    velocity_base: int = 80
    enabled: bool = True


@dataclass
class CorpusConfig:
    """Configuration for corpus ingestion."""
    data_dir: str = "data"
    quantize_resolution: int = 4
    transpose_to_c: bool = True
    context_length: int = 16


@dataclass
class GuideRenderingConfig:
    """Configuration for guide-audio rendering."""
    enabled: bool = False
    sample_rate: int = 44100
    output_stems: bool = True
    output_phrases: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for structural evaluation."""
    enabled: bool = True
    run_challenge_sets: bool = False


@dataclass
class GenerationConfig:
    """Master configuration for the entire generation pipeline."""
    # Legacy components
    tonal: TonalConfig = field(default_factory=TonalConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    gttm: GTTMConfig = field(default_factory=GTTMConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    # New components (revised strategy)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    realizer: RealizerConfig = field(default_factory=RealizerConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    guide_rendering: GuideRenderingConfig = field(default_factory=GuideRenderingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    genre: str = "prog_rock"
    seed: Optional[int] = None
    device: str = "cpu"
    output_dir: str = "output"
    verbose: bool = True

    @classmethod
    def from_json(cls, path: str) -> "GenerationConfig":
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                if hasattr(section, "__dataclass_fields__"):
                    for k, v in section_data.items():
                        if hasattr(section, k):
                            setattr(section, k, v)
                else:
                    setattr(config, section_name, section_data)
        return config

    def to_json(self, path: str):
        import dataclasses
        data = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if dataclasses.is_dataclass(val):
                data[f.name] = dataclasses.asdict(val)
            else:
                data[f.name] = val
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as fp:
            json.dump(data, fp, indent=2)

    @classmethod
    def prog_rock(cls) -> "GenerationConfig":
        """Preset for progressive rock generation."""
        config = cls(genre="prog_rock")
        config.structure.bpm = 110.0
        config.structure.time_signatures = [(4, 4), (7, 8), (5, 4), (6, 8)]
        config.structure.phrase_types = [
            "intro", "theme_a", "theme_b", "development",
            "solo", "recapitulation", "coda"
        ]
        config.gttm.prolongational_weight = 1.5
        config.gttm.tension_curve_resolution = 32
        return config

    @classmethod
    def jazz_fusion(cls) -> "GenerationConfig":
        """Preset for jazz fusion generation."""
        config = cls(genre="jazz_fusion")
        config.structure.bpm = 130.0
        config.structure.time_signatures = [(4, 4), (7, 4), (5, 4)]
        config.structure.phrase_types = [
            "head_in", "solo_section", "interlude",
            "trading_fours", "head_out", "coda"
        ]
        config.gttm.prolongational_weight = 1.2
        config.tonal.is_major = True
        return config
