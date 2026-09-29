"""Small helper for pickle-backed caches that follow a shared shape:
load on startup, rebuild when missing/corrupt, cheap staleness check
via a separate `fingerprint` field so callers don't have to unpickle
large payloads just to compare mtimes."""
import pickle
from pathlib import Path

from analysis import cache_dir


class Cache:
    def __init__(self, filename):
        self.path = cache_dir() / filename

    def _read(self):
        if not self.path.exists():
            return None
        try:
            with open(self.path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def load(self):
        """Return stored data, or None if missing/corrupt."""
        blob = self._read()
        if blob is None:
            return None
        return blob.get('data')

    def load_fresh(self, fingerprint):
        """Stored data if it was saved under `fingerprint`, else None.
        One read, where `fingerprint()` then `load()` costs two."""
        blob = self._read()
        if blob is None or blob.get('fingerprint') != fingerprint:
            return None
        return blob.get('data')

    def fingerprint(self):
        """Return stored fingerprint without touching data. Still pays an
        unpickle cost ; Python's pickle format is not random-access ; but
        callers can use it to decide whether to discard `load()`'s result."""
        blob = self._read()
        if blob is None:
            return None
        return blob.get('fingerprint')

    def save(self, data, fingerprint=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, 'wb') as f:
                pickle.dump({'fingerprint': fingerprint, 'data': data}, f)
        except OSError:
            pass

    def clear(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
