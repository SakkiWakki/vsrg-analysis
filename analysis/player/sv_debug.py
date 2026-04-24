"""Opt-in JSONL logging for SV/render debugging."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class SVDebugLogger:
    def __init__(self) -> None:
        self._enabled = os.environ.get('VSRG_SV_DEBUG', '').strip().lower() in {
            '1', 'true', 'yes', 'on',
        }
        path = os.environ.get('VSRG_SV_DEBUG_LOG', '/tmp/vsrg_sv_debug.jsonl')
        self._path = Path(path)
        self._every = max(1, int(os.environ.get('VSRG_SV_DEBUG_EVERY', '1')))
        raw_max = int(os.environ.get('VSRG_SV_DEBUG_MAX_FRAMES', '2000'))
        self._max_frames = None if raw_max == 0 else max(1, raw_max)
        self._frame_count = 0
        self._logged = 0
        self._lock = threading.Lock()
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, record: dict) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._frame_count += 1
            if self._max_frames is not None and self._logged >= self._max_frames:
                return
            if (self._frame_count - 1) % self._every != 0:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open('a', encoding='utf-8') as f:
                if not self._started:
                    f.write(json.dumps({
                        'type': 'session_start',
                        'every': self._every,
                        'max_frames': self._max_frames,
                    }, sort_keys=True) + '\n')
                    self._started = True
                f.write(json.dumps(record, sort_keys=True) + '\n')
            self._logged += 1


LOGGER = SVDebugLogger()
