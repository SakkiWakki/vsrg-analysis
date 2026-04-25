"""Dedicated PortAudio stream-owner thread.

Holds the OutputStream's lifecycle (create / start / stop / close) on a
background thread instead of the Qt main thread. The stream's internal
callback thread can otherwise inherit GIL/scheduling characteristics
from whichever Python thread opened it; on PulseAudio/PipeWire this
shows up as the audio callback fighting the GUI thread for the GIL
during long paints, audible as choppiness.

The worker doesn't process anything itself -- the callback still runs
on PortAudio's own thread. It only owns the stream object, so all
stream-state transitions happen from one consistent thread.
"""
from __future__ import annotations

import threading


class StreamWorker:
    """Owns one PortAudio OutputStream on a background thread.

    Usage:
        worker = StreamWorker()
        stream, err = worker.open(make_stream)
        # ... use stream from any thread (callback is internal to it) ...
        worker.close(stream)

    `make_stream` is a callable returning a started OutputStream. It runs
    on the worker thread; if it raises, the exception is propagated back
    via the second tuple element.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def open(self, make_stream):
        """Run `make_stream()` on the worker thread; return `(stream, err)`.

        Blocks the caller until the stream is created (success or error).
        On success the worker thread parks until `close()` is called, so
        the stream stays bound to a consistent thread context for its
        lifetime.
        """
        result = {'stream': None, 'err': None}
        ready = threading.Event()

        def run():
            try:
                result['stream'] = make_stream()
            except Exception as e:
                result['err'] = e
            finally:
                ready.set()
            self._stop.wait()

        self._thread = threading.Thread(
            target=run, daemon=True, name='vsrg-audio-stream',
        )
        self._thread.start()
        ready.wait()
        return result['stream'], result['err']

    def close(self, stream) -> None:
        """Stop + close the stream and let the worker thread exit.

        Safe to call when no stream was ever opened (`open()` returned
        an error) -- in that case the worker thread is already parked
        on `_stop` waiting to exit, and `stream` is None.
        """
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
