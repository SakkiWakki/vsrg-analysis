"""LRU-bounded parsed-replay cache. Keyed by (replay_path, chart_path) so
multiple viz tabs / player openings over the same replay don't re-parse the
.osr/.osu file each time."""
from analysis.gui.loaders import load_replay


class ReplayCache:
    def __init__(self, max_size=16):
        self._cache = {}
        self._order = []
        self._max = max_size

    def get(self, entry):
        key = (entry.get('replay_path'), entry.get('chart_path'))
        hit = self._cache.get(key)
        if hit is not None:
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return hit
        rep = load_replay(entry['replay_path'], entry['game'],
                          entry.get('chart_path'))
        self._cache[key] = rep
        self._order.append(key)
        while len(self._order) > self._max:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return rep
