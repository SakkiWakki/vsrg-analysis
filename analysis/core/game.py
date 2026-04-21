"""Per-game adapter base class + dynamic discovery.

Each subdirectory of `analysis/games/` may contain an `adapter.py` that
defines a `GameAdapter` subclass named `ADAPTER`. `discover_games()` scans
the directory at import time (or on first `get()` call) and populates the
registry. Each adapter is also responsible for registering its scroll
modes via `analysis/player/scroll.py` at import time.

See analysis/games/README.md for the full contract.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


class GameAdapter:
    name: str = ''

    def parse_replay(self, path, chart_path=None):
        raise NotImplementedError

    def resolve_audio(self, replay, entry=None, progress=None) -> str | None:
        return None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        """Return (bpms, sm_offset). Both may be None/0.0 if not applicable."""
        return None, 0.0

    def judgement_windows(self, replay, **overrides):
        raise NotImplementedError

    def judge_label(self, replay, **overrides) -> str:
        raise NotImplementedError

    def default_scroll_mode(self) -> str:
        return 'ms'

    def player_kwargs(self, replay, **overrides) -> dict:
        """Extra kwargs the Player __init__ needs for this game
        (OD for osu, judge for etterna)."""
        return {}

    def prepare_replay_times(self, replay, **timing):
        """Return (times_sec, hold_tails, keycount) for this replay.
        times_sec is a float64 array parallel to replay['noterows'];
        hold_tails maps (noterow, column) -> tail time in seconds.
        Per-game because osu replays store absolute ms while Etterna
        stores SM noterows that need the chart's BPM map to time out."""
        raise NotImplementedError

    def judge_kwarg_name(self) -> str:
        """Name of the keyword the game's judge system takes in
        `judgement_windows`/`judge_label`/`player_kwargs` ('judge' for
        Etterna, 'od' for osu). The Player uses this to forward the
        active value without branching per game."""
        return 'judge'

    def nudge_judge(self, current, delta):
        """Step the current judge value by `delta`. Signed; sign is all
        that matters for discrete judges (Etterna J1..J9), magnitude
        matters for continuous ones (osu OD float).

        Returns the new value, or `current` if the adapter doesn't
        support switching (default). Called from the sidebar ± buttons
        and the Player's keyboard shortcut path."""
        return current


_REGISTRY: dict[str, GameAdapter] = {}
_discovered = False


def discover_games() -> None:
    """Scan `analysis/games/` for adapter modules. Each game package is
    expected to expose `ADAPTER` (an instance of a `GameAdapter` subclass)
    from its `adapter` module. Idempotent."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    import analysis.games as games_pkg
    games_dir = Path(games_pkg.__file__).parent
    for info in pkgutil.iter_modules([str(games_dir)]):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(f'analysis.games.{info.name}.adapter')
        except ModuleNotFoundError:
            continue
        adapter = getattr(mod, 'ADAPTER', None)
        if isinstance(adapter, GameAdapter):
            _REGISTRY[adapter.name or info.name] = adapter


def get(name: str) -> GameAdapter:
    if not _discovered:
        discover_games()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown game: {name!r}')


def all_games() -> dict[str, GameAdapter]:
    if not _discovered:
        discover_games()
    return dict(_REGISTRY)
