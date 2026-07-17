"""Compile-time samplers for gat's per-frame AFT-copy drivers.

gat's field copies (`gat_aft_target`, ...) are positioned two ways:

- one-shot `mod_message` tweens that poke the copy directly - these are
  already recorded onto the copy's timeline by the mod-actions replay
  (modfile._field_copies merges them in); and

- per-frame `UpdateCommand` closures that read HIDDEN DATA-HOLDER QUADS
  (`gat_aftx`, `gat_afty`, `gat_aftzoom`, `gat_aftrz`) and write the copy
  each frame, e.g. (default.xml, the `perframe(1140,1146)` driver):

      gat_aft_target:zoom(gat_aftzoom:GetX())
      gat_aft_target:y(SCREEN_CENTER_Y + gat_afty:GetY() + gat_afty2:GetY())
      gat_aft_target:x(SCREEN_CENTER_X + gat_aftx:GetX())
      gat_aft_target:rotationz(gat_aftrz:GetX())

  The driver expressions are pure functions of the (already compiled)
  quad timelines, so we grid-sample the closure at compile: at each grid
  time in the beat window we read the source quad timelines and emit a
  keyframe on the copy. The copy's transform then translates/zooms/
  rotates exactly as the live per-frame driver would, with no runtime
  interpretation. This is the "grid-sample the deterministic drivers at
  compile against the compiled quad curves" plan (scoping item 25).

A driver is declared once per (copy, window) with the source quads and
a closure computing (x, y, rotation, scale_x, scale_y) from the sampled
quad values. Only drivers whose source quads actually compiled are
emitted, so a chart lacking them is unaffected.
"""
from __future__ import annotations

from analysis.player.render.effects.timeline import Keyframe

_SCREEN_CENTER_X = 320.0
_SCREEN_CENTER_Y = 240.0

# Grid resolution for sampling a driver window: seconds between samples.
# 1/60s tracks the source quad eases finely enough that the copy's motion
# reads as continuous (the quad tweens are >= ~0.1s long).
_GRID_STEP_S = 1.0 / 60.0
_DRIVEN_PROPS = ('x', 'y', 'rotation', 'scale_x', 'scale_y')


def _q(sources, name, prop, t):
    """Sample source quad `name`'s `prop` timeline at `t`, or 0 when the
    quad (or that prop) never compiled."""
    quad = sources.get(name)
    if quad is None:
        return 0.0
    timeline = quad.get(prop)
    return timeline.sample(t)[0] if timeline is not None else 0.0


def _gat_aft_target_driver(sources, t):
    """perframe(1140,1146): gat_aft_target driven by the aft quads.
    x/y are screen-center + quad offset; zoom is uniform from aftzoom;
    rotationz from aftrz. (effectmagnitude/vib is a jitter we skip -
    it has no still-frame analogue.)"""
    zoom = _q(sources, 'gat_aftzoom', 'x', t)
    return {
        'x': _SCREEN_CENTER_X + _q(sources, 'gat_aftx', 'x', t),
        'y': (_SCREEN_CENTER_Y + _q(sources, 'gat_afty', 'y', t)
              + _q(sources, 'gat_afty2', 'y', t)),
        'rotation': _q(sources, 'gat_aftrz', 'x', t),
        'scale_x': zoom,
        'scale_y': zoom,
    }


# (copy actor global -> (source quad names, window (t_start, t_end),
# closure)). Windows are seconds; the caller supplies the beat->seconds
# map so the declaration stays in the template's beat space.
_DRIVERS = {
    'gat_aft_target': {
        'sources': ('gat_aftx', 'gat_afty', 'gat_afty2', 'gat_aftzoom',
                    'gat_aftrz'),
        'beats': (1140.0, 1146.0),
        'closure': _gat_aft_target_driver,
    },
}


def has_driver(copy_name) -> bool:
    return copy_name in _DRIVERS


def driven_keyframes(copy_name, source_timelines, to_seconds) -> dict:
    """Grid-sampled keyframes for a driven copy, as {prop: [Keyframe]}
    over the driver window, or {} when the copy has no driver / its
    source quads never compiled.

    `source_timelines` maps a quad name to its {prop: EventTimeline}
    dict (the compiled data-holder curves). Each emitted keyframe is a
    zero-duration hold at a grid point; a fine grid makes the staircase
    read as the continuous per-frame driver."""
    driver = _DRIVERS.get(copy_name)
    if driver is None:
        return {}
    sources = {name: source_timelines.get(name) for name in driver['sources']}
    if not any(sources.values()):
        return {}

    t0 = to_seconds(driver['beats'][0])
    t1 = to_seconds(driver['beats'][1])
    closure = driver['closure']
    out = {prop: [] for prop in _DRIVEN_PROPS}
    for t in _grid(t0, t1):
        values = closure(sources, t)
        for prop in _DRIVEN_PROPS:
            out[prop].append(Keyframe(t=t, values=(values[prop],),
                                     duration=0.0, easing=0))
    return {prop: frames for prop, frames in out.items() if frames}


def _grid(t0, t1):
    if t1 <= t0:
        return
    t = t0
    while t < t1:
        yield t
        t += _GRID_STEP_S
    yield t1
