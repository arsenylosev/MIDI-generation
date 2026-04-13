"""Audio rendering modules.

Provides two rendering strategies:

    1. **AudioRenderer** (legacy) — Single-mix rendering via FluidSynth or
       pure-Python fallback.  Useful for quick previews.

    2. **GuideAudioRenderer** (recommended) — Per-stem guide channel rendering
       aligned to phrase boundaries.  Designed for feeding into contemporary
       music audio models as conditioning signals (revised strategy note,
       Sections 7–8).
"""

from midi_gen.rendering.audio_renderer import AudioRenderer, render_midi_to_audio
from midi_gen.rendering.guide_renderer import GuideAudioRenderer, GuideBundle, PhraseGuide

__all__ = [
    "AudioRenderer",
    "render_midi_to_audio",
    "GuideAudioRenderer",
    "GuideBundle",
    "PhraseGuide",
]
