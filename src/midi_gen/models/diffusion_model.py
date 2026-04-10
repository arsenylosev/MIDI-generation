"""
Lightweight Diffusion Model for MIDI Piano-Roll Generation.

This module implements a simplified latent diffusion model that generates
piano-roll representations of music. It is designed to be compatible with
the whole-song-gen architecture while being lightweight enough for
demonstration and rapid prototyping.

The model operates on piano-roll tensors of shape (2, T, 128) where:
- Channel 0: onset events
- Channel 1: sustain events
- T: number of time steps (16th note resolution)
- 128: MIDI pitch range
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model."""
    num_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"  # "linear" or "cosine"
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4


class NoiseSchedule:
    """Manages the noise schedule for the diffusion process."""

    def __init__(self, config: DiffusionConfig):
        self.num_steps = config.num_steps

        if config.schedule == "linear":
            self.betas = np.linspace(config.beta_start, config.beta_end, config.num_steps)
        elif config.schedule == "cosine":
            steps = np.arange(config.num_steps + 1) / config.num_steps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0: np.ndarray, t: int,
                  noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = np.random.randn(*x0.shape)
        noisy = (self.sqrt_alpha_bars[t] * x0 +
                 self.sqrt_one_minus_alpha_bars[t] * noise)
        return noisy, noise

    def remove_noise(self, xt: np.ndarray, predicted_noise: np.ndarray,
                     t: int) -> np.ndarray:
        """Remove predicted noise from xt at timestep t."""
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        mean = (1.0 / np.sqrt(alpha)) * (
            xt - (beta / self.sqrt_one_minus_alpha_bars[t]) * predicted_noise
        )

        if t > 0:
            noise = np.random.randn(*xt.shape)
            sigma = np.sqrt(beta)
            return mean + sigma * noise
        return mean


class SimpleUNet:
    """
    Simplified U-Net-like architecture for noise prediction.

    This is a numpy-based implementation for demonstration purposes.
    For production use, this should be replaced with a PyTorch/JAX
    implementation with proper training.
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.rng = np.random.default_rng(42)

        # Initialize weights for a simple MLP-based denoiser
        self.layers = self._init_layers()

    def _init_layers(self) -> List[dict]:
        """Initialize layer weights with Xavier initialization."""
        layers = []
        dims = [256 + 1]  # input dim (flattened) + time embedding

        for i in range(self.config.num_layers):
            in_dim = dims[-1]
            out_dim = self.config.hidden_dim
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            layers.append({
                "W": self.rng.normal(0, scale, (in_dim, out_dim)),
                "b": np.zeros(out_dim),
            })
            dims.append(out_dim)

        # Output layer
        in_dim = dims[-1]
        out_dim = 256  # same as input
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        layers.append({
            "W": self.rng.normal(0, scale, (in_dim, out_dim)),
            "b": np.zeros(out_dim),
        })

        return layers

    def predict_noise(self, xt: np.ndarray, t: int,
                      condition: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict the noise component in xt at timestep t.

        For the demo, this uses a simple forward pass through the MLP.
        In production, this would be a trained U-Net.
        """
        original_shape = xt.shape
        x = xt.flatten()

        # Truncate or pad to expected dimension
        target_dim = 256
        if len(x) > target_dim:
            x = x[:target_dim]
        elif len(x) < target_dim:
            x = np.pad(x, (0, target_dim - len(x)))

        # Time embedding (sinusoidal)
        t_emb = np.array([t / self.config.num_steps])
        x = np.concatenate([x, t_emb])

        # Forward pass through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = x @ layer["W"][:len(x)] + layer["b"] if len(x) == layer["W"].shape[0] else layer["b"]
            x = np.maximum(x, 0.01 * x)  # LeakyReLU

        # Output layer
        out_layer = self.layers[-1]
        if len(x) == out_layer["W"].shape[0]:
            x = x @ out_layer["W"] + out_layer["b"]
        else:
            x = out_layer["b"]

        # Reshape to original
        result = np.zeros(np.prod(original_shape))
        result[:min(len(x), len(result))] = x[:min(len(x), len(result))]
        return result.reshape(original_shape)


class MidiDiffusionModel:
    """
    Main diffusion model for MIDI piano-roll generation.

    Combines the noise schedule, U-Net denoiser, and conditioning
    mechanisms to generate piano-roll representations.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.schedule = NoiseSchedule(self.config)
        self.denoiser = SimpleUNet(self.config)

    def generate(
        self,
        num_steps: int,
        condition: Optional[np.ndarray] = None,
        tension_curve: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate a piano-roll using the reverse diffusion process.

        Args:
            num_steps: Number of time steps in the output
            condition: Optional conditioning signal (e.g., form structure)
            tension_curve: Optional tension curve to guide generation
            rng: Random number generator

        Returns:
            Piano-roll array of shape (2, num_steps, 128)
        """
        if rng is None:
            rng = np.random.default_rng()

        shape = (2, num_steps, 128)
        xt = rng.normal(0, 1, shape)

        # Reverse diffusion
        for t in reversed(range(self.config.num_steps)):
            predicted_noise = self.denoiser.predict_noise(xt, t, condition)

            # Apply tension-guided scaling if available
            if tension_curve is not None:
                tension_scale = np.interp(
                    np.arange(num_steps),
                    np.linspace(0, num_steps - 1, len(tension_curve)),
                    tension_curve
                )
                # Modulate noise prediction by tension
                for step_idx in range(num_steps):
                    predicted_noise[:, step_idx, :] *= (0.5 + tension_scale[step_idx])

            xt = self.schedule.remove_noise(xt, predicted_noise, t)

        # Post-process: threshold to binary piano-roll
        piano_roll = self._post_process(xt)
        return piano_roll

    def _post_process(self, raw_output: np.ndarray) -> np.ndarray:
        """
        Post-process raw diffusion output to a valid piano-roll.

        Applies thresholding, ensures mutual exclusivity of onset/sustain,
        and enforces basic musical constraints.
        """
        # Sigmoid-like activation
        output = 1.0 / (1.0 + np.exp(-raw_output))

        # Threshold
        onset = (output[0] > 0.5).astype(float)
        sustain = (output[1] > 0.3).astype(float)

        # Ensure sustain only follows onset
        for p in range(128):
            in_note = False
            for t in range(onset.shape[0]):
                if onset[t, p] > 0:
                    in_note = True
                    sustain[t, p] = 0  # No sustain on onset
                elif sustain[t, p] > 0 and not in_note:
                    sustain[t, p] = 0  # Remove orphan sustain
                elif sustain[t, p] == 0:
                    in_note = False

        return np.stack([onset, sustain], axis=0)

    def generate_conditioned(
        self,
        phrase_structure: list,
        key: int = 0,
        is_major: bool = True,
        beats_per_measure: int = 4,
        steps_per_beat: int = 4,
        tension_curve: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate a conditioned piano-roll based on phrase structure.

        This is the primary interface for the generation pipeline.
        """
        if rng is None:
            rng = np.random.default_rng()

        total_measures = sum(p["lgth"] for p in phrase_structure)
        num_steps = total_measures * beats_per_measure * steps_per_beat

        # Create conditioning signal from phrase structure and key
        condition = self._encode_condition(
            phrase_structure, key, is_major,
            beats_per_measure, steps_per_beat
        )

        return self.generate(num_steps, condition, tension_curve, rng)

    def _encode_condition(
        self,
        phrase_structure: list,
        key: int,
        is_major: bool,
        beats_per_measure: int,
        steps_per_beat: int
    ) -> np.ndarray:
        """Encode musical conditions into a conditioning vector."""
        total_measures = sum(p["lgth"] for p in phrase_structure)
        total_steps = total_measures * beats_per_measure * steps_per_beat

        # Create a per-step conditioning matrix
        condition = np.zeros((total_steps, 16))

        # Key encoding (one-hot for 12 pitch classes + major/minor)
        condition[:, key % 12] = 1.0
        condition[:, 12] = 1.0 if is_major else 0.0

        # Phrase type encoding
        step = 0
        for phrase in phrase_structure:
            phrase_steps = phrase["lgth"] * beats_per_measure * steps_per_beat
            phrase_type_code = hash(phrase.get("name", "A")) % 3
            condition[step:step + phrase_steps, 13] = phrase_type_code / 3.0
            # Relative position within phrase
            for t in range(phrase_steps):
                condition[step + t, 14] = t / max(1, phrase_steps - 1)
            step += phrase_steps

        return condition
