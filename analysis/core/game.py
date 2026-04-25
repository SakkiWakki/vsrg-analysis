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

    def build_sv_engine(self, replay):
        """Build an SVEngine for this replay, or return None for identity SV.

        Returning None is NOT the same as raising NotImplementedError: it
        means "this chart has no scroll-velocity data", and the Player runs
        with distance(a, b) = b - a. Override for games that model scroll
        velocity (osu!mania timing points, Etterna #SCROLLS/#SPEEDS, etc.)."""
        return None

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

    # --- library scan -----------------------------------------------------
    def scan_library(self, progress=None) -> list:
        """Return a list of entry dicts for every playable replay on disk.
        Must include at minimum `game`, `replay_path`, `mtime`. Called by
        `analysis.core.search.build_library()`."""
        return []

    # --- library cache lifecycle -----------------------------------------
    # Three entry points, all optional. Default impls fall back to
    # `scan_library` so an adapter that only implements the old hook still
    # works ; it just won't benefit from incremental updates or separate
    # caching.
    def load_cached(self) -> list | None:
        """Return this game's cached entries, or None if no valid cache
        exists. Cheap: should not do a full rescan."""
        return None

    def save_cached(self, entries: list) -> None:
        """Persist this game's entries to the adapter's cache. Called
        when a consumer (e.g. the GUI) mutates entries in place and
        wants the change to survive across runs."""
        pass

    def incremental_update(self, progress=None) -> list:
        """Fast path: return the complete entry list for this game after
        picking up any new replays since the last rebuild. If there's no
        cache yet, behaves like `rebuild`."""
        return self.rebuild(progress=progress)

    def rebuild(self, progress=None) -> list:
        """Slow path: wipe this game's caches, rescan everything, write a
        fresh cache, and return the entry list."""
        return self.scan_library(progress=progress) or []

    # --- standalone-launch resolver (CLI / player __main__) ---------------
    def can_handle_path(self, path) -> bool:
        """True if `path` looks like a replay this adapter can parse ; used
        by the player's standalone entry point to pick an adapter."""
        return False

    def resolve_standalone(self, path, args=None):
        """Parse a replay + resolve audio + return (replay, bpms, sm_offset,
        audio, extra_kwargs) for the standalone player entry point. `args`
        is the raw argv after the replay path so adapters can pick up
        optional flags (e.g. --sm, --osu, --bpm)."""
        raise NotImplementedError

    # --- PlayerTab construction -------------------------------------------
    def player_tab_kwargs(self, replay, entry, chart_ctx) -> dict:
        """Extra keyword arguments for `PlayerTab.__init__` beyond the shared
        set (game, audio_path, scroll_ms, scroll_mode, play_rate).
        `chart_ctx` is the (bpms, sm_offset, audio) tuple the GUI adapter's
        `resolve_chart_context` returned."""
        return {}

    # --- per-note-sprite rasterize overrides ------------------------------
    def note_sprites(self, replay) -> dict:
        """Return `{sprite_name: SpriteSpec}` for this replay. Each spec
        declares pixmap size + a rasterize callback + key fields. The
        renderer allocates pixmaps lazily on first draw of each distinct
        key combination, so games that never produce a note type (e.g.
        osu has no mines) pay zero memory for it.

        Default returns the baseline skin; override to replace any
        entry, or omit keys to disable a sprite entirely."""
        from analysis.player.render.layers.note_sprites import (
            default_note_sprites)
        return default_note_sprites()

    # --- per-game note-type declarations ----------------------------------
    def note_types(self, replay) -> list:
        """Return the list of `NoteType` the renderer should draw for
        this replay. Each entry becomes its own toggleable layer in the
        HUD. Adapters can subset, reorder, or extend the default set ;
        a game without mines simply omits the `mines` entry so no dead
        toggle appears. Default: every note type the shared renderer
        layers know how to draw."""
        from analysis.player.render.layers.notes import default_note_types
        return default_note_types()

    # --- note visualizer windows ------------------------------------------
    def viz_windows(self, replay, judge=None, od=None):
        """Return (windows, unit_label, rows_per_ms) for the note visualizer.
        Used by render_chart_full / interactive in analysis/viz/note_visualizer.
        `windows` is the list-of-tuples shape that viz expects (name, w_s, color)."""
        raise NotImplementedError


def resolve_standalone_replay(path, args=None):
    """Pick the adapter that claims `path`, run its standalone resolver.
    Returns (game_name, replay, bpms, sm_offset, audio, extra_kwargs)."""
    for name, adapter in all_games().items():
        if adapter.can_handle_path(path):
            rep, bpms, off, audio, extra = adapter.resolve_standalone(
                path, args=args)
            return name, rep, bpms, off, audio, extra
    raise ValueError(f'no adapter claims replay path: {path!r}')


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
