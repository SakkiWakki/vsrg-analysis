"""Compile-side composition: one transform channel per field instance.

Every field instance - the two player fields and every proxy/AFT copy -
is drawn by the renderer as one blit of a playfield capture under one
transform. This module builds that transform at compile time from the
recorded per-actor poke streams; the consumer
(field_instances.NotitgFieldInstances) samples `.at(t)` and draws, with
no per-property knowledge.

Engine transform model (openitg Actor::BeginDraw + RageDisplay's matrix
stack, MultMatrixLocal): an actor's pushes apply to its content
innermost-first as

    skew_x -> rotation XYZ (fused) -> scale -> translate

and a child's matrix composes onto its parent's. The composed world
matrix projects through the screen perspective (RageDisplay
LoadMenuPerspective: fov 45, centered vanish), so out-of-plane tilts
(rotation_x/rotation_y) become true perspective while purely 2D
transforms reduce to the exact affine map. Order matters visibly: a
flip (negative scale) after a spin mirrors the spin direction, and a
skew inside a rotation shears along the rotated axis - both are chart
staples.

Capture mapping: the renderer's field capture holds the playfield
rendered at rest - design-space content centered on the design center -
so the channel conjugates the world matrix with T(-center) on the
content side. An instance sitting at the design center with unit scale
is the identity. The consumer conjugates the sampled homography by the
chart-region design map to reach window space.

Channels are live-evaluated curves over the recorded EventTimelines
(events-not-keyframes): nothing here is pre-sampled per frame.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render import transform3d
from analysis.player.render.effects.timeline import EventTimeline

_DESIGN_W = 640.0
_DESIGN_H = 480.0
_CENTER_X = _DESIGN_W / 2.0
_CENTER_Y = _DESIGN_H / 2.0
_CORNERS = ((0.0, 0.0), (_DESIGN_W, 0.0),
            (_DESIGN_W, _DESIGN_H), (0.0, _DESIGN_H))
_TO_CONTENT = transform3d.translate(-_CENTER_X, -_CENTER_Y)

# RageDisplay LoadMenuPerspective defaults: fov 45, vanish at the screen
# centre. (Recorded per-proxy vanish-point channels exist in the
# compiled dict but are not consumed yet.)
_PROJECTION = transform3d.projection(45.0, _DESIGN_W, _DESIGN_H, vanish=None)

_MIN_ALPHA = 1.0 / 255.0
_MIN_DET = 1e-9

# Transform properties every link carries, with SM rest values.
_LINK_RESTS = {
    'x': 0.0, 'y': 0.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0, 'skew_x': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0,
    'alpha': 1.0, 'hidden': 0.0,
}

# Player-field rest seats in design space: StepMania places each versus
# player at the PlayerP{n}X metric (center -+ 160), SCREEN_CENTER_Y
# (openitg ScreenGameplay player placement). Recorded pokes are seeded
# from these, so the rests are only a backstop for never-poked players.
PLAYER_REST = {
    1: {'x': _CENTER_X - 160.0, 'y': _CENTER_Y},
    2: {'x': _CENTER_X + 160.0, 'y': _CENTER_Y},
}


def link_timelines(keyframes, rests=None) -> dict:
    """One sampleable timeline per link property from an actor's
    recorded keyframes; missing properties rest (`rests` overrides
    per-property rest values, e.g. the player seats)."""
    merged = {**_LINK_RESTS, **(rests or {})}
    keyframes = keyframes or {}
    return {prop: EventTimeline(keyframes.get(prop, []), rest=(rest,))
            for prop, rest in merged.items()}


class _SumTimeline:
    """Additive overlay: oscillator deltas riding on a recorded stream."""

    def __init__(self, timelines):
        self._timelines = tuple(timelines)

    def sample(self, t):
        return (sum(tl.sample(t)[0] for tl in self._timelines),)


def overlay_deltas(link, deltas) -> dict:
    """The link with delta timelines (field-oscillator x/y/rotation)
    summed onto the matching properties."""
    out = dict(link)
    for prop, delta in (deltas or {}).items():
        out[prop] = _SumTimeline((out[prop], delta))
    return out


class TransformChannel:
    """The composed capture->design transform of one field instance.

    `links` are the instance's actors root-first, each a full
    `link_timelines` dict. `t0` clamps sampling to the compile start, so
    pre-chart times hold the load state instead of pre-load rests (a
    proxy resting `hidden,1` from its InitCommand must not flash visible
    before the first beat). `flip_base_y` negates the leaf's
    base_scale_y: AFT samplers set basezoomy(-1) to compensate the
    engine's bottom-up GL captures, and our capture is already top-down,
    so the sign convention is translated (a chart leaving basezoomy at
    +1 to want the raw flipped texture still lands).

    `.at(t)` returns `(H, alpha)` - H the 3x3 row-vector homography
    mapping capture coords onto the design screen - or None when the
    instance is invisible: hidden anywhere in the chain, alpha ~0,
    degenerate scale, or tilted fully behind the eye."""

    def __init__(self, links, t0=None, flip_base_y=False):
        self._links = tuple(links)
        self._t0 = t0
        self._flip_base_y = flip_base_y

    def at(self, t):
        if self._t0 is not None:
            t = max(float(t), self._t0)
        alpha = 1.0
        world = None
        leaf = len(self._links) - 1
        for i, link in enumerate(self._links):
            if link['hidden'].sample(t)[0] >= 0.5:
                return None
            alpha *= link['alpha'].sample(t)[0]
            local = self._local(link, t, self._flip_base_y and i == leaf)
            world = local if world is None else transform3d.compose(world,
                                                                    local)
        if alpha < _MIN_ALPHA:
            return None
        verdict, H, _clip = transform3d.project_with_verdict(
            _TO_CONTENT @ world, _PROJECTION, _CORNERS)
        if verdict == 'gone' or abs(np.linalg.det(H)) < _MIN_DET:
            return None
        return H, alpha

    @staticmethod
    def _local(link, t, flip):
        def v(prop):
            return link[prop].sample(t)[0]

        base_sy = v('base_scale_y')
        if flip:
            base_sy = -base_sy
        m = transform3d.rotate_xyz(v('rotation_x'), v('rotation_y'),
                                   v('rotation'))
        m = m @ transform3d.scale(v('scale_x') * v('base_scale_x'),
                                  v('scale_y') * base_sy)
        m = m @ transform3d.translate(v('x'), v('y'))
        skew = v('skew_x')
        if skew:
            m = transform3d.skew_x(skew) @ m
        return m


def instance(name, kind, player, links, t0=None, aft_order=None,
             aft_live=None, color=None) -> dict:
    """One compiled field instance. `kind` is 'player' (a real player's
    always-rendered field group), 'proxy' (an ActorProxy re-render of
    player `player`'s notefield), 'aft' (an ActorFrameTexture screen
    sampler), or 'fill' (an AFT-rig curtain quad blitted at its tree
    position); `player` is the 1-based player whose capture the instance
    blits (0 for 'aft'/'fill'). `aft_order` places an 'aft' sampler
    relative to its source AFT node in draw order: 'post' samplers show
    the frame's fresh capture (drawn after the node captured), 'pre'
    samplers show the previous frame's (their draw preceded this frame's
    capture). `aft_live` samples the source node's visibility (0.0 =
    hidden = the preserve-texture capture is frozen); `color` samples a
    'fill' curtain's diffuse rgb."""
    return {'name': name, 'kind': kind, 'player': player,
            'aft_order': aft_order, 'aft_live': aft_live, 'color': color,
            'transform': TransformChannel(links, t0=t0,
                                          flip_base_y=kind == 'aft')}


def player_link(number, keyframes, osc_deltas=None,
                ignore_hidden=False) -> dict:
    """A player field's own transform link: the recorded PlayerP{n} poke
    stream (may be empty: an untouched player rests at its versus seat)
    with its field-oscillator deltas overlaid. `ignore_hidden` pins the
    hidden channel to visible - a proxy draws its target WITH the
    target's transform but REGARDLESS of the target's hidden bit (the
    standard trick: hide the real field, let the proxies show it)."""
    link = overlay_deltas(link_timelines(keyframes, rests=PLAYER_REST[number]),
                          osc_deltas)
    if ignore_hidden:
        link['hidden'] = EventTimeline([], rest=(0.0,))
    return link


def player_instance(number, keyframes, osc_deltas=None, t0=None) -> dict:
    """A player-field instance drawn in place from its own link."""
    link = player_link(number, keyframes, osc_deltas)
    return instance(f'P{number}', 'player', number, [link], t0=t0)


# Proxy source tags -> the player whose notefield they re-render
# (ScreenGameplay names players 1-based; odd tags target player 1).
_PROXY_PLAYERS = {'P1p': 1, 'P3p': 1, 'P2p': 2, 'P4p': 2}


def harvest_instances(field_copies, player_keyframes=None,
                      field_oscillators=None, dual=False) -> list:
    """The generic instance list from a harvest-path compiled dict
    (single-link 'field_copies' + the player poke streams). The
    engine-loop compiler emits 'field_instances' directly; this fallback
    keeps the frozen harvest path on the same consumer contract."""
    instances = []
    for copy in field_copies or ():
        player = _PROXY_PLAYERS.get(copy['source'], 0)
        kind = 'proxy' if player else 'aft'
        link = {**link_timelines(None), **copy['timelines']}
        # The harvest dict carries no draw-order info; 'post' (fresh
        # capture) is the common sampler shape.
        instances.append(instance(copy['name'], kind, player, [link],
                                  aft_order='post' if kind == 'aft' else None))
    if dual:
        keyframes = player_keyframes or {}
        oscillators = field_oscillators or {}
        for number in (1, 2):
            instances.append(player_instance(number,
                                             keyframes.get(f'P{number}'),
                                             oscillators.get(number)))
    return instances
