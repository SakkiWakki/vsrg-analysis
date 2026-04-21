"""Rate-aware streaming audio engine for the replay player.

Chain (modeled on the IMS pull-based `generate` pattern):

    WaveSource  →  StreamingPhaseVocoder  →  sounddevice callback

The callback runs on the audio thread at ~5ms cadence and pulls N frames
from the top of the chain. Because every effect works per-buffer, rate
changes and pitch-correct toggles are instant. `AudioEngine.set_state(t,
rate, playing)` is the only public surface PlayerTab touches."""
from .engine import AudioEngine
from .phase_vocoder import StreamingPhaseVocoder
from .source import WaveSource

__all__ = ['AudioEngine', 'StreamingPhaseVocoder', 'WaveSource']
