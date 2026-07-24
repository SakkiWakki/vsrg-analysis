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

from analysis.games.notitg import field_projection
from analysis.player.render import transform3d
from analysis.player.render.effects.timeline import EventTimeline

_CENTER_X = field_projection.DESIGN_CX
_CENTER_Y = field_projection.DESIGN_CY
_CORNERS = field_projection.PLANE_CORNERS
_TO_CONTENT = transform3d.translate(-_CENTER_X, -_CENTER_Y)

_MIN_ALPHA = 1.0 / 255.0
_MIN_DET = 1e-9
_REST_EPS = 1e-4

# Transform properties every link carries, with SM rest values. `fov`
# is the frame camera's field of view (deg): not a transform matrix
# term (`_local` never reads it) but a perspective-projection parameter
# an ActorFrame sets for its whole subtree - its rest is the
# LoadMenuPerspective default the shared projection uses.
_LINK_RESTS = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0,
    'skew_x': 0.0, 'skew_y': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0, 'base_scale_z': 1.0,
    'alpha': 1.0, 'hidden': 0.0,
    'fov': field_projection.FOV,
    # Fork transform-order state (NotITG SetRotationOrder + skew-before
    # gates): 'skew_*_before' rest at 0 (skew applies AFTER rotation, the
    # engine default), so an untouched link keeps the stock compose.
    'skew_x_before': 0.0, 'skew_y_before': 0.0,
    # Texture-edge insets (SetCrop*, fraction hidden per edge) and anchor
    # fractions (SetHorizAlign/SetVertAlign, 0.5 = centered): the AFT
    # band idiom crops half the sampler and re-anchors the survivor in
    # place. Crop is consumed by `crop_at` (a source clip on the blit);
    # align by the leaf's local matrix.
    'crop_top': 0.0, 'crop_bottom': 0.0, 'crop_left': 0.0, 'crop_right': 0.0,
    'halign': 0.5, 'valign': 0.5,
}

_CROP_PROPS = ('crop_left', 'crop_top', 'crop_right', 'crop_bottom')

# Non-scalar link channels sampled as whole tuples, not per-component
# curves: the rotation-order token ('xyz'..) and the accumulated spherical
# quaternion (x, y, z, w). Both write as immediate keyframes, so the
# EventTimeline holds the last value with no interpolation, and both rest
# at the engine default (stock order, identity quat) - an untouched link
# composes byte-identically to the pre-order path.
_STOCK_ROTATION_ORDER = transform3d._ROTATION_ORDERS[0]
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_TUPLE_LINK_RESTS = {
    'rotation_order': (_STOCK_ROTATION_ORDER,),
    'quat': _IDENTITY_QUAT,
}

# Player-field rest seats in design space: StepMania places the two
# versus players at the PlayerP{n}X metric (center -+ 160),
# SCREEN_CENTER_Y (openitg ScreenGameplay player placement). Recorded
# pokes are seeded from these, so the rests are only a backstop for
# never-poked players. Extra players (P3-P8 - the SRT charts' decorative
# field SOURCES) have no style seat; they rest at center and the chart's
# own pokes place them.
_PLAYER_SEATS = {
    1: {'x': _CENTER_X - 160.0, 'y': _CENTER_Y},
    2: {'x': _CENTER_X + 160.0, 'y': _CENTER_Y},
}


def player_rest(number):
    return _PLAYER_SEATS.get(number, {'x': _CENTER_X, 'y': _CENTER_Y})


def link_timelines(keyframes, rests=None) -> dict:
    """One sampleable timeline per link property from an actor's
    recorded keyframes; missing properties rest (`rests` overrides
    per-property rest values, e.g. the player seats)."""
    merged = {**_LINK_RESTS, **(rests or {})}
    keyframes = keyframes or {}
    timelines = {prop: EventTimeline(keyframes.get(prop, []), rest=(rest,))
                 for prop, rest in merged.items()}
    for prop, rest in _TUPLE_LINK_RESTS.items():
        timelines[prop] = EventTimeline(keyframes.get(prop, []), rest=rest)
    return timelines


def link_live_timelines(sim, rec_id, rests=None) -> dict:
    """LAZY counterpart of `link_timelines`: one `LiveCurve` per link property,
    reading the live sim's actor at draw time (same key set + rests, so
    `TransformChannel` samples it identically). Covers the scalar link props and
    the tuple ones (rotation_order token, quat)."""
    from analysis.games.notitg.sim.seg_read import curve_for

    merged = {**_LINK_RESTS, **(rests or {})}
    timelines = {prop: curve_for(sim, rec_id, prop, rest)
                 for prop, rest in merged.items()}
    for prop, rest in _TUPLE_LINK_RESTS.items():
        timelines[prop] = curve_for(sim, rec_id, prop, rest)
    return timelines


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
        fov = field_projection.FOV
        leaf = len(self._links) - 1
        for i, link in enumerate(self._links):
            if link['hidden'].sample(t)[0] >= 0.5:
                return None
            alpha *= link['alpha'].sample(t)[0]
            # A frame's fov projects its whole subtree; the innermost
            # frame that set one wins (its LoadMenuPerspective replaces
            # the outer). Links are root-first, so the last non-default
            # fov in the chain is the effective camera.
            link_fov = link['fov'].sample(t)[0]
            if abs(link_fov - field_projection.FOV) > _REST_EPS:
                fov = link_fov
            local = self._local(link, t, self._flip_base_y and i == leaf,
                                i == leaf)
            world = local if world is None else transform3d.compose(world,
                                                                    local)
        if alpha < _MIN_ALPHA:
            return None
        projection = field_projection.design_projection(fov=fov)
        verdict, H, _clip = transform3d.project_with_verdict(
            _TO_CONTENT @ world, projection, _CORNERS)
        if verdict == 'gone' or abs(np.linalg.det(H)) < _MIN_DET:
            return None
        return H, alpha

    def crop_at(self, t):
        """The instance's crop insets `(left, top, right, bottom)` as
        fractions of its texture, or None at rest (no crop - today's
        exact blit). Crop is read from the LEAF link only: SetCrop*
        insets the drawn quad of the actor that owns a texture, and only
        the leaf sprite has one (an ActorFrame's crop is a no-op)."""
        if self._t0 is not None:
            t = max(float(t), self._t0)
        link = self._links[-1]
        crop = tuple(link[prop].sample(t)[0] for prop in _CROP_PROPS)
        if all(edge <= _REST_EPS for edge in crop):
            return None
        return crop

    @staticmethod
    def _local(link, t, flip, leaf):
        def v(prop):
            return link[prop].sample(t)[0]

        # Anchor offset (halign/valign): the engine offsets the QUAD's
        # vertices, innermost of everything, so it rides every later
        # zoom/rotation. Leaf-only: only the sprite has content to
        # anchor. Centered (0.5) is the rest identity.
        adx = ady = 0.0
        if leaf:
            adx = (0.5 - v('halign')) * field_projection.DESIGN_W
            ady = (0.5 - v('valign')) * field_projection.DESIGN_H

        base_sy = v('base_scale_y')
        if flip:
            base_sy = -base_sy
        rx, ry, rz = v('rotation_x'), v('rotation_y'), v('rotation')
        sx = v('scale_x') * v('base_scale_x')
        sy = v('scale_y') * base_sy
        sz = v('scale_z') * v('base_scale_z')
        skew = v('skew_x')
        skewy = v('skew_y')
        z = v('z')
        quat = link['quat'].sample(t)
        has_quat = quat != _IDENTITY_QUAT
        if (not (rx or ry or rz or skew or skewy or z or has_quat)
                and sx == 1.0 and sy == 1.0 and sz == 1.0):
            # The overwhelmingly common link state (a plain positioned
            # frame): one translation matrix instead of three matmuls,
            # sampled for every instance link every frame.
            return transform3d.translate(v('x') + adx, v('y') + ady)
        (order,) = link['rotation_order'].sample(t)
        m = transform3d.rotate_ordered(rx, ry, rz, order)
        if has_quat:
            # Spherical adds (heading/pitch/roll) ride a quaternion the
            # engine composes just after the Euler rotation (Actor.cpp
            # BeginDraw:424-429): content rotates, then the quat spins it.
            m = m @ transform3d.matrix_from_quat(quat)
        m = m @ transform3d.scale(sx, sy, sz)
        m = m @ transform3d.translate(v('x'), v('y'), z)
        # skew_*_before_rotation toggles which side of the rotation the
        # skew composes on (fork BeginDraw skew-order gate). Flag 0 is the
        # stock placement this module has always used (skew applied to
        # content before the rotate/scale block); the flag flips it to the
        # far side, so an untouched link is byte-identical to before.
        if skew:
            if v('skew_x_before') >= 0.5:
                m = m @ transform3d.skew_x(skew)
            else:
                m = transform3d.skew_x(skew) @ m
        if skewy:
            if v('skew_y_before') >= 0.5:
                m = m @ transform3d.skew_y(skewy)
            else:
                m = transform3d.skew_y(skewy) @ m
        if adx or ady:
            m = transform3d.translate(adx, ady) @ m
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
    link = overlay_deltas(link_timelines(keyframes, rests=player_rest(number)),
                          osc_deltas)
    if ignore_hidden:
        link['hidden'] = EventTimeline([], rest=(0.0,))
    return link


def player_live_link(sim, number, rec_id, osc_deltas=None,
                     ignore_hidden=False) -> dict:
    """LAZY player-field link: LiveCurves over the live PlayerP{n} actor
    (`rec_id`), seated at its versus rest. Same overlay/ignore_hidden as
    `player_link`."""
    link = overlay_deltas(
        link_live_timelines(sim, rec_id, rests=player_rest(number)),
        osc_deltas)
    if ignore_hidden:
        link['hidden'] = EventTimeline([], rest=(0.0,))
    return link


def player_instance(number, keyframes, osc_deltas=None, t0=None) -> dict:
    """A player-field instance drawn in place from its own link."""
    link = player_link(number, keyframes, osc_deltas)
    return instance(f'P{number}', 'player', number, [link], t0=t0)


def player_live_instance(sim, number, rec_id, osc_deltas=None, t0=None) -> dict:
    """LAZY player-field instance: LiveCurves over the live PlayerP{n} actor."""
    link = player_live_link(sim, number, rec_id, osc_deltas)
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
