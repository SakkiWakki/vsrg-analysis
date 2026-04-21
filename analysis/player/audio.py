"""Rate-aware audio engine for the replay player.

Single-input design: callers drive the engine by calling `set_state(t, rate,
playing)` every tick. The engine is responsible for deciding whether it needs
to restart its internal sound (rate changed, big time jump, pause toggled,
audio finished) — callers never touch pygame directly.

Rate handling is resample-only: pitch shifts with rate, no time-stretch.
Matches Etterna/osu!mania's native rate-mod feel without a pitch-correction
plugin."""
from __future__ import annotations
import os
import time
import numpy as np
import pygame


class AudioEngine:
    # If the chart time and where the audio thinks it is diverge by more than
    # this, we restart the underlying sound at the new position instead of
    # letting it drift.
    RESYNC_THRESHOLD_S = 0.15

    def __init__(self, audio_path: str | None, volume: float = 0.5):
        self.ready = False
        self._volume = float(volume)
        self._base = None        # int16 samples, shape (n,) or (n, 2)
        self._sr = None          # Hz
        self._channels = 1
        self._channel = None     # pygame.mixer.Channel
        self._active_sound = None  # pygame.Sound currently playing
        self._anchor_t = 0.0       # chart time at which the current sound was started
        self._anchor_wall = None   # monotonic time of start
        self._rate = 1.0           # rate the current sound was built at
        # Cache the resampled int16 arrays directly (one per rate). We used to
        # cache pygame.Sound objects, but playing from the middle still forced
        # an array→Sound roundtrip per seek; caching the array lets us build a
        # sliced Sound with a single copy on resume.
        self._arr_cache: dict[float, np.ndarray] = {}
        self._ended = False
        # Background precompute of common rates so hitting rate+/rate- doesn't
        # stall the UI the first time each one is used.
        self._prewarm_thread = None

        self._base_duration = 0.0
        if not audio_path or not os.path.exists(audio_path):
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            raw = pygame.mixer.Sound(audio_path)
            arr = pygame.sndarray.array(raw)
            self._base = arr.astype(np.int16, copy=False)
            init = pygame.mixer.get_init()
            self._sr = int(init[0]) if init else 44100
            self._channels = 1 if arr.ndim == 1 else int(arr.shape[1])
            self._base_duration = float(raw.get_length())
            self.ready = True
        except Exception as e:
            print(f'audio: {e}')
            self.ready = False

    # -------- internals --------
    def _build_array_for_rate(self, rate: float) -> np.ndarray | None:
        """Return the full resampled int16 array for `rate`, cached.

        Uses integer-space arithmetic for indexing (avoids allocating a
        float64 array the size of the song), which is the main bottleneck
        when switching rates on long tracks."""
        if not self.ready:
            return None
        key = round(rate, 3)
        cached = self._arr_cache.get(key)
        if cached is not None:
            return cached
        if abs(rate - 1.0) < 1e-3:
            out = self._base
        else:
            n = self._base.shape[0]
            new_n = max(1, int(n / rate))
            # Build index in int64 space at a fixed-point resolution of 1<<16
            # so we don't materialize a giant float64 array.
            step = int(rate * (1 << 16))
            idx = (np.arange(new_n, dtype=np.int64) * step) >> 16
            np.clip(idx, 0, n - 1, out=idx)
            out = self._base[idx]
            if not out.flags['C_CONTIGUOUS']:
                out = np.ascontiguousarray(out)
        self._arr_cache[key] = out
        return out

    def prewarm_rates(self, rates) -> None:
        """Precompute resampled arrays for the given rates on a background
        thread so the first use of each doesn't stall the UI."""
        if not self.ready:
            return
        import threading
        if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
            return

        def run():
            for r in rates:
                try:
                    self._build_array_for_rate(float(r))
                except Exception:
                    pass

        self._prewarm_thread = threading.Thread(
            target=run, name='audio-prewarm', daemon=True)
        self._prewarm_thread.start()

    def _start_from(self, t: float, rate: float) -> None:
        """(Re)start playback from chart time `t` at the given rate."""
        self._stop()
        arr = self._build_array_for_rate(rate)
        if arr is None:
            return
        t = max(0.0, float(t))
        start_sec = t / max(0.05, rate)     # position inside the rate-adjusted sound
        skip = int(start_sec * self._sr)
        if skip >= len(arr):
            return
        # Slice directly from the cached numpy array and build ONE Sound —
        # no intermediate sndarray.array() copy, no extra allocations.
        if skip > 0:
            sliced = arr[skip:]
            if not sliced.flags['C_CONTIGUOUS']:
                sliced = np.ascontiguousarray(sliced)
        else:
            sliced = arr
        try:
            play_snd = pygame.sndarray.make_sound(sliced)
            play_snd.set_volume(self._volume)
            ch = play_snd.play()
        except Exception:
            ch = None
        if ch is None:
            return
        self._channel = ch
        self._active_sound = play_snd
        self._anchor_t = t
        self._anchor_wall = _monotonic()
        self._rate = rate

    def _stop(self) -> None:
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        self._channel = None
        self._active_sound = None
        self._anchor_wall = None

    def _predicted_t(self) -> float | None:
        """Where the audio currently is (in chart time), or None if stopped."""
        if self._channel is None or self._anchor_wall is None:
            return None
        elapsed = _monotonic() - self._anchor_wall
        return self._anchor_t + elapsed * self._rate

    # -------- public API --------
    def set_volume(self, v: float) -> None:
        self._volume = float(v)
        if self._active_sound is not None:
            try: self._active_sound.set_volume(self._volume)
            except Exception: pass

    def set_state(self, t: float, rate: float, playing: bool) -> None:
        """Single entry point. Feed the current (t, rate, playing) every tick
        and the engine keeps itself in sync."""
        if not self.ready:
            return
        rate = max(0.05, float(rate))

        if not playing:
            self._stop()
            return

        # If chart time is past the end of the audio, don't restart — just
        # stay quiet. Prevents looping the last few seconds of the file.
        if t >= self._base_duration:
            self._stop()
            self._ended = True
            return
        self._ended = False

        # Need to (re)start if: nothing is playing, rate changed, or audio
        # position has drifted from chart time beyond the threshold.
        restart = False
        if self._channel is None:
            restart = True
        elif abs(rate - self._rate) > 1e-3:
            restart = True
        else:
            pred = self._predicted_t()
            if pred is None or abs(pred - t) > self.RESYNC_THRESHOLD_S:
                restart = True
            elif self._channel is not None and not self._channel.get_busy():
                # Sound finished naturally at the true end of the file —
                # don't restart; let the GUI observe `ended` and stop playback.
                self._channel = None
                self._ended = True
                restart = False

        if restart:
            self._start_from(t, rate)

    def stop(self) -> None:
        self._stop()


def _monotonic() -> float:
    return time.monotonic()
