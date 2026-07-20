"""Keyframe-parity diff: do two sim runs animate every actor the same?

The engine loop can be driven two ways - the chart's own Lua (today) or the
AST interpreter (`frame_eval`, incoming) - both poking the SAME `SimActor`s
through the SAME scheduler. Correctness of the interpreter is therefore
exactly: does the recorded surface it produces MATCH the one the Lua path
produces? This module answers that, and is the acceptance oracle the design
calls for ("compiled must pass the same tests the sim does").

It compares by PLAYED-BACK VALUE, not by keyframe identity. Two runs may
record a property as different keyframe shapes - a run of instants vs one
collapsed tween (`simplify_instants`), an immediate write vs a zero-tween -
yet play back the same curve. What a viewer sees is `EventTimeline.sample(t)`,
so that is what we diff: sample both timelines on a shared grid and flag a
time where the values differ beyond tolerance. A representation difference
that plays identically is NOT a divergence (it is the same animation), which
is the whole point of recording compact.

A divergence names the actor (its source-XML label, stable across runs), the
property, the time, and both values - so "map X plays wrong" becomes "actor
FOO.rotation diverges at t=12.3: lua=90.0 interp=0.0", a diagnosable gap
rather than a montage mystery.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.games.notitg.lua_api import _REST
from analysis.player.render.effects.timeline import EventTimeline

# Per-property closeness in the property's own unit. Positions/rotations are
# design pixels / degrees (ENGINE_ORACLE trace tolerances); alpha and the
# 0..1 fades are fractions; everything else takes the position bar. A diff
# under this at every sampled time is the same animation.
_POSITION_TOL = 1.0
_FRACTION_TOL = 0.01
_FRACTION_PROPS = frozenset({
    'alpha', 'fade_left', 'fade_right', 'fade_top', 'fade_bottom',
    'crop_top', 'crop_bottom', 'crop_left', 'crop_right'})


def _tol(prop: str) -> float:
    return _FRACTION_TOL if prop in _FRACTION_PROPS else _POSITION_TOL


@dataclass(frozen=True)
class Divergence:
    """One (actor, property) mismatch at one sampled time."""
    actor: str
    prop: str
    t: float
    left: tuple
    right: tuple

    def __str__(self) -> str:
        return (f'{self.actor}.{self.prop} @ t={self.t:.4f}: '
                f'left={_fmt(self.left)} right={_fmt(self.right)}')


def _fmt(values: tuple) -> str:
    return '(' + ', '.join(f'{v:.4g}' if isinstance(v, (int, float))
                           else repr(v) for v in values) + ')'


def diff_runs(left_env, right_env, times,
              labels=None) -> list[Divergence]:
    """Every (actor, property, time) where `left_env` and `right_env` play
    back different values, over sample `times` (song seconds).

    Actors are joined by LABEL (source-XML `file:Name`), which is stable
    across two runs of the same chart even though recorder ids need not be;
    `labels` overrides the join map (rec_id -> label) when a caller has a
    better one (e.g. bound-global names). A property recorded by only one
    run is compared against the other's REST value, so a spurious poke on one
    side surfaces as a divergence from rest rather than being silently
    dropped."""
    left = _by_label(left_env, labels)
    right = _by_label(right_env, labels)
    out = []
    for label in sorted(set(left) | set(right)):
        out.extend(_diff_actor(label, left.get(label, {}),
                               right.get(label, {}), times))
    return out


def _diff_actor(label, left_props, right_props, times) -> list[Divergence]:
    out = []
    for prop in sorted(set(left_props) | set(right_props)):
        rest = (_rest(prop),)
        lt = EventTimeline(left_props.get(prop, []), rest)
        rt = EventTimeline(right_props.get(prop, []), rest)
        tol = _tol(prop)
        for t in times:
            lv = lt.sample(t)
            rv = rt.sample(t)
            if not _close(lv, rv, tol):
                out.append(Divergence(label, prop, float(t), lv, rv))
    return out


def _close(left: tuple, right: tuple, tol: float) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a - b) > tol:
                return False
        elif a != b:
            return False
    return True


def _rest(prop: str) -> float:
    value = _REST.get(prop)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _by_label(env, labels) -> dict:
    """{actor label: {property: [Keyframe]}} for one run. The label join is
    what lets two runs (different rec-id assignment order) line up: an actor's
    `file:Name` is a chart property, its recorder id is a run artifact."""
    id_to_label = dict(labels) if labels is not None else _env_labels(env)
    out: dict = {}
    for rec_id, frames in env.actor_keyframes().items():
        label = id_to_label.get(rec_id, f'actor#{rec_id}')
        # Two actors can share a label only if the chart gave two nodes the
        # same Name in the same file; merge their streams so neither is lost.
        merged = out.setdefault(label, {})
        for prop, kfs in frames.items():
            merged.setdefault(prop, []).extend(kfs)
    return out


def _env_labels(env) -> dict:
    """rec_id -> a CROSS-RUN-STABLE label for the diff join. Priority:
      1. a real source `file:Name` (chart-intrinsic, no `#rec_id` suffix);
      2. a bound-global name (`named_actor_ids`, stable tiebreak);
      3. the actor's structural tree path (`actor_tree_paths`).
    A raw `kind#rec_id` label or the `actor#rec_id` fallback is NEVER used to
    join two runs - the rec_id is a run artifact, so those buckets would not
    line up (two runs assign ids in different order). The tree path is the
    stable identity for the anonymous actors that dominate a real chart."""
    raw = dict(getattr(env, '_labels', {}))
    named = env.named_actor_ids()
    paths = env.actor_tree_paths() if hasattr(env, 'actor_tree_paths') else {}
    labels = {}
    for rec_id in set(raw) | set(named) | set(paths):
        source = raw.get(rec_id, '')
        if source and '#' not in source:
            labels[rec_id] = source          # real file:Name
        elif rec_id in named:
            labels[rec_id] = named[rec_id]    # bound-global name
        elif rec_id in paths:
            labels[rec_id] = f'@{paths[rec_id]}'  # structural tree path
        else:
            labels[rec_id] = source or f'actor#{rec_id}'
    return labels


def sample_grid(t_start: float, t_end: float, hz: float = 60.0) -> list:
    """A dense song-time sample grid for `diff_runs`. Matching the sim tick
    rate means a per-frame driver's every step is checked."""
    step = 1.0 / hz
    times = []
    t = float(t_start)
    while t <= t_end:
        times.append(t)
        t += step
    return times
