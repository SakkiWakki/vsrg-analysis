"""Per-column note-path splines + arrowpath gradient (NotITG fork).

`Player:SetXSpline(point, col, value_px, y_offset_domain, mode)` (and
SetZSpline) shape a per-column displacement curve over the note path:
ArrowEffects::GetXPos / GetZPos (@004ef8e0 / @004f7020) sample the
column's spline at each arrow's y-offset and ADD the value to its X / Z
position, so notes, receptors (y-offset 0), hold-body strips, and the
drawn arrowpath all follow one curve (W32_Stuxnet's helix: 11 points,
domain 0..1000 in 100px steps, X=sin / Z=cos).

`SetNumPathGradientPoints(col, n)` / `SetPathGradientColor(point, col,
r, g, b, a)` color the drawn arrowpath (the `arrowpath` mod's trail)
with up to n gradient stops along that same domain.

The sim records each write onto the PLAYER actor as instant channels
(actor.py `_poke_note_path`):

    spline_x:{col}:{idx} / spline_z:{col}:{idx} -> (value, domain)
    pathgrad:{col}:{idx}                        -> (r, g, b, a)
    pathgrad_n:{col}                            -> count

This module rebuilds the sampled curves from those channels:
`build_players(env)` -> {player0: PlayerNotePath}, whose `sampler_at(t)`
yields the per-frame `SplineSampler` note_mods folds into its offsets.

Interpolation: `sample_note_path` is a non-uniform Catmull-Rom (central-
difference tangents, one-sided at the ends), clamped beyond the domain.
The fork's exact kernel is COMDAT-folded in the decompile
(Player.clean.c aliases the spline thunks), so the standard smooth
spline is used; the reference helix is visibly smooth, ruling out
linear. Revisit only against captured frames.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.effects.timeline import EventTimeline

_ACTIVE_EPS = 1e-4

# A spline-point channel before its first poke: no contribution. The
# NaN domain marks the point unset so the sampler drops it (a 0 value
# at domain 0 would instead pin every curve's head to the column rest).
_POINT_REST = (0.0, float('nan'))
_GRAD_REST = (1.0, 1.0, 1.0, 1.0)

_SPLINE_PREFIXES = {'spline_x:': 'x', 'spline_z:': 'z'}


def sample_note_path(domains, values, queries):
    """Sample the control polygon (domains ascending, one value each) at
    `queries` with a non-uniform Catmull-Rom: cubic Hermite per segment,
    central-difference tangents (`np.gradient` over the domain spacing),
    one-sided at the first/last point, queries clamped to the domain.
    One point degenerates to a constant, two to the linear chord."""
    d = np.asarray(domains, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    if len(d) == 1:
        return np.full(q.shape, v[0])
    q = np.clip(q, d[0], d[-1])
    k = np.clip(np.searchsorted(d, q, side='right') - 1, 0, len(d) - 2)
    h = d[k + 1] - d[k]
    safe_h = np.where(h > 0.0, h, 1.0)
    f = np.where(h > 0.0, (q - d[k]) / safe_h, 0.0)
    slopes = np.gradient(v, d) if len(d) > 2 else \
        np.full(2, (v[1] - v[0]) / max(d[1] - d[0], 1e-9))
    m0 = slopes[k] * h
    m1 = slopes[k + 1] * h
    f2 = f * f
    f3 = f2 * f
    return ((2.0 * f3 - 3.0 * f2 + 1.0) * v[k] + (f3 - 2.0 * f2 + f) * m0
            + (-2.0 * f3 + 3.0 * f2) * v[k + 1] + (f3 - f2) * m1)


class SplineSampler:
    """One frame's spline curves: (axis, col) -> (domains, values), only
    columns whose curve is live at this instant. Built by
    `PlayerNotePath.sampler_at`; consumed by note_mods (notes, hold-body
    strips, receptors) and the arrowpath ribbon."""

    __slots__ = ('_curves',)

    def __init__(self, curves: dict):
        self._curves = curves

    def columns(self, axis: str) -> tuple:
        return tuple(sorted(col for ax, col in self._curves if ax == axis))

    def offsets(self, axis: str, cols, y_offsets) -> np.ndarray:
        """Per-note spline displacement (engine px) for note columns
        `cols` at scroll offsets `y_offsets`; columns without a live
        curve contribute 0."""
        cols = np.asarray(cols)
        y_offsets = np.asarray(y_offsets, dtype=np.float64)
        out = np.zeros(y_offsets.shape, dtype=np.float64)
        for col in np.unique(cols):
            curve = self._curves.get((axis, int(col)))
            if curve is None:
                continue
            mask = cols == col
            out[mask] = sample_note_path(curve[0], curve[1],
                                         y_offsets[mask])
        return out

    def column_curve(self, axis: str, col: int):
        """(domains, values) for one column, or None."""
        return self._curves.get((axis, int(col)))


class PlayerNotePath:
    """One player's recorded note-path state: spline point timelines per
    (axis, col, idx) plus the arrowpath gradient stops."""

    __slots__ = ('_points', '_grad_colors', '_grad_counts')

    def __init__(self, points, grad_colors, grad_counts):
        self._points = points          # (axis, col) -> ((idx, tl), ...)
        self._grad_colors = grad_colors  # col -> ((idx, tl), ...)
        self._grad_counts = grad_counts  # col -> tl

    def sampler_at(self, t: float) -> SplineSampler | None:
        """The live spline curves at `t`, or None when every point is
        unset or ~0 (the inert fast path note_mods keys off)."""
        curves = {}
        live = False
        for key, points in self._points.items():
            domains, values = [], []
            for _idx, timeline in points:
                value, domain = timeline.sample(t)
                if not np.isfinite(domain):
                    continue
                domains.append(domain)
                values.append(value)
            if not domains:
                continue
            order = np.argsort(np.asarray(domains), kind='stable')
            d = np.asarray(domains, dtype=np.float64)[order]
            v = np.asarray(values, dtype=np.float64)[order]
            curves[key] = (d, v)
            live = live or bool(np.max(np.abs(v)) >= _ACTIVE_EPS)
        return SplineSampler(curves) if live else None

    def gradient_at(self, t: float, col: int) -> list:
        """The arrowpath gradient stops for `col` at `t`: [(r, g, b, a),
        ...] in point order, truncated to the recorded stop count
        (default 1 stop, white)."""
        stops = self._grad_colors.get(col, ())
        count_tl = self._grad_counts.get(col)
        count = int(count_tl.sample(t)[0]) if count_tl is not None else 1
        colors = [tl.sample(t) for _idx, tl in stops]
        if not colors:
            return [_GRAD_REST]
        return colors[:max(1, count)]


class NotePathHandle:
    """The compiled document's note-path surface: 0-based player ->
    PlayerNotePath. Starts empty on the lazy path and is `swap`ped in by
    the background sweep (the ScrollMultiplierHandle hot-swap shape)."""

    __slots__ = ('_players',)

    def __init__(self, players: dict | None = None):
        self._players = players or {}

    def __bool__(self):
        return bool(self._players)

    def swap(self, players: dict) -> None:
        self._players = players

    def player(self, player: int) -> PlayerNotePath | None:
        return self._players.get(player)

    def sampler_at(self, t: float, player: int) -> SplineSampler | None:
        path = self._players.get(player)
        return path.sampler_at(t) if path is not None else None


def build_players(env) -> dict:
    """0-based player -> PlayerNotePath for every player actor whose
    recorded channels carry spline / arrowpath-gradient pokes."""
    players = {}
    for number in range(1, 9):
        actor = env.player_actor(f'PlayerP{number}')
        if actor is None:
            continue
        path = _build_one(actor.keyframes())
        if path is not None:
            players[number - 1] = path
    return players


def _build_one(keyframes: dict) -> PlayerNotePath | None:
    points: dict = {}
    grad_colors: dict = {}
    grad_counts: dict = {}
    for prop, kfs in keyframes.items():
        spline = _spline_key(prop)
        if spline is not None:
            points.setdefault(spline[:2], []).append(
                (spline[2], EventTimeline(kfs, rest=_POINT_REST)))
            continue
        if prop.startswith('pathgrad_n:'):
            grad_counts[int(prop.rsplit(':', 1)[1])] = EventTimeline(
                kfs, rest=(1.0,))
        elif prop.startswith('pathgrad:'):
            _tag, col, idx = prop.split(':')
            grad_colors.setdefault(int(col), []).append(
                (int(idx), EventTimeline(kfs, rest=_GRAD_REST)))
    if not points and not grad_colors and not grad_counts:
        return None
    return PlayerNotePath(
        {key: tuple(sorted(pts)) for key, pts in points.items()},
        {col: tuple(sorted(stops)) for col, stops in grad_colors.items()},
        grad_counts)


def _spline_key(prop: str):
    """('x'|'z', col, idx) for a spline point channel name, else None."""
    for prefix, axis in _SPLINE_PREFIXES.items():
        if prop.startswith(prefix):
            col, idx = prop[len(prefix):].split(':')
            return (axis, int(col), int(idx))
    return None
