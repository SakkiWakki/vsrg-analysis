"""Fullscreen shader stack sampled from `.ffx`-style shader events.

Each event eases three params (`strength`..`strength3`) for one named
shader from the previous event's target (or the event's own
`start-params` when `use-start` is set) to `end-params` over
`[time, time + duration]`; params rest at 0 and a shader is active
only while any param is positive. Shaders keep first-appearance order,
matching fluXis's `ShaderStackContainer` build order.

Pure sampling only: the per-frame result is `(shader_id, uniforms)`
pairs on `EffectFrame.shaders`, executed by the GL pipeline
(render/shaders/gl_pipeline.py). Ids are lowercased event names;
unknown ids are carried through and skipped at execution.

Not fluXis-specific: any game emitting keyframed fullscreen-shader
streams maps them onto this effect.
"""
from __future__ import annotations

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

_REST = (0.0, 0.0, 0.0)


def _params(raw) -> tuple:
    raw = raw if isinstance(raw, dict) else {}
    return tuple(float(raw.get(key, 0.0) or 0.0)
                 for key in ('strength', 'strength2', 'strength3'))


def _keyframe(event) -> Keyframe:
    return Keyframe(
        t=float(event.get('time', 0.0)) / 1000.0,
        values=_params(event.get('end-params') or event.get('params')),
        duration=max(0.0, float(event.get('duration', 0.0) or 0.0)) / 1000.0,
        easing=int(event.get('ease', 0) or 0),
        start=(_params(event.get('start-params'))
               if event.get('use-start') else None),
    )


class ShaderStackEffect:
    """Samples one `EventTimeline` per shader name and emits the
    active passes for the frame."""

    def __init__(self, events):
        grouped: dict[str, list] = {}
        for event in events or []:
            if not isinstance(event, dict):
                continue
            name = str(event.get('shader', '')).strip().lower()
            if not name:
                continue
            grouped.setdefault(name, []).append(_keyframe(event))
        self._stacks = tuple(
            (name, EventTimeline(keyframes, rest=_REST))
            for name, keyframes in grouped.items())

    def __bool__(self):
        return bool(self._stacks)

    def at(self, ctx) -> EffectFrame | None:
        passes = []
        for name, timeline in self._stacks:
            strength = timeline.sample(ctx.t_now)
            if any(v > 0.0 for v in strength):
                passes.append((name, {'u_strength': strength}))
        if not passes:
            return None
        return EffectFrame(shaders=tuple(passes))
