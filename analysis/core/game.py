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
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of an attempted game launch.

    ``ok=True`` means the game subprocess started; the overlay-injection
    side-effect happens later (e.g. Windows DLL injection runs after a
    delay) so a True result doesn't guarantee the overlay is visible
    yet, only that the launch path didn't refuse upfront. Callers
    surface ``message`` to the user when ``ok`` is false.

    ``path_label`` names the launch strategy that was picked
    ('gl-layer' / 'vulkan-layer' / 'gamescope' / 'win-gl-layer'); useful
    for diagnostic logging. ``extra`` is for adapter-specific bookkeeping
    callers shouldn't need to interpret.
    """
    ok: bool
    message: str = ''
    pid: int | None = None
    path_label: str = ''
    extra: dict = field(default_factory=dict)


def _game_packages():
    """Names of every package under `analysis.games/` (one per game).
    Shared by `GameAdapter` and `GameManifest` discovery so adding a game
    means one new directory, not two parallel scanners."""
    import analysis.games as games_pkg
    games_dir = Path(games_pkg.__file__).parent
    return [info.name for info in pkgutil.iter_modules([str(games_dir)])
            if info.ispkg]


def _load_adapters(submodule: str, attr: str, base_cls) -> dict:
    """Import `analysis.games.<pkg>.<submodule>` for each game package and
    collect its `<attr>` attribute when it's an instance of `base_cls`.
    Used by both the `GameAdapter` and `GameManifest` registries."""
    out: dict = {}
    for pkg in _game_packages():
        try:
            mod = importlib.import_module(f'analysis.games.{pkg}.{submodule}')
        except ModuleNotFoundError:
            continue
        adapter = getattr(mod, attr, None)
        if isinstance(adapter, base_cls):
            out[adapter.name or pkg] = adapter
    return out


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

    def lane_mask(self, replay):
        """Active-lane timeline `[(t_start_s, mask_tuple, dur_s, easing)]`
        for charts whose visible lane set changes mid-play (fluXis lane
        switches), or None for static layouts. Masks are per-lane 0/1
        tuples of length keycount; note arrays stay full-width with
        absolute columns, so only lane geometry consults this."""
        return None

    def background_path(self, replay) -> str | None:
        """Absolute path to this replay's map background image, or None
        when the game/chart has none. Rendered behind the playfield
        (dimmed, cover-fit) by the builtin background effect. Resolved
        like `resolve_audio`: a filename from the chart, joined to the
        chart's folder, existence-checked."""
        return None

    def effects(self, replay) -> list:
        """Column-space visual effects for this replay (playfield
        transforms, storyboards, ...); see
        `analysis.player.render.effects`. Default: none. The lane-mask
        collapse is provided as an effect automatically when
        `lane_mask()` is non-None, so adapters only override this for
        extra effects."""
        return []

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

    def viz_panel_units(self, replay) -> int:
        """Default panel/window size for the note visualizer, in this
        adapter's `viz_windows` units (noterows for Etterna, ms for the
        time-space games). 2400 is the SM/Etterna 8-measure default;
        ms-axis games override since 2400 ms is too narrow."""
        return 2400

    def populate_notes_model(self, replay, model) -> None:
        """Fill any per-game extras on the NotesModel beyond the shared
        noterow/column/LN bookkeeping. osu pulls ghost-tap and miss-hold
        spans off the replay; Etterna copies mines/lifts/fakes/rolls from
        the matched chart. Default: nothing extra."""
        return None

    # --- launching the game ----------------------------------------------
    def launch(self, *, with_overlay: bool = True):
        """Start the game with the host's overlay attached, if supported.

        Returns a :class:`LaunchResult` describing the outcome.
        Adapters that don't support being launched from the host
        (e.g. Etterna) raise :class:`NotImplementedError`. Callers
        branch on ``LaunchResult.ok`` and surface ``message`` to the
        user when false; this base does not show dialogs or otherwise
        touch UI.
        """
        raise NotImplementedError(
            f'{self.name!r} does not support launching from the host')

    def judgment_colors(self) -> dict:
        """RGB tuples for each judgment-window name this adapter produces.
        The renderer indexes into this by the label `judge()` returns, so a
        new window name needs an entry here or the renderer KeyErrors at
        draw time. Default covers the union across builtin games; override
        to add or recolor windows."""
        from analysis.player.init.judgment import JCLR
        return JCLR


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
    _REGISTRY.update(_load_adapters('adapter', 'ADAPTER', GameAdapter))


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
