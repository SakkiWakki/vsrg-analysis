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

from analysis.games.notitg import field_projection, mod_channels
from analysis.player.render import transform3d
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

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
    # Fork Player hibernate gate (SetAwake @0x00533780): an asleep
    # player neither updates nor draws. Rest 1 (born awake), so only a
    # chart's explicit SetAwake(false) blanks its field instance.
    'awake': 1.0,
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
    # Face culling (SetCullMode): 0 none, 1 back, 2 front. Leaf-only, judged
    # on the ENGINE winding - see TransformChannel.at.
    'cull': 0.0,
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
    the tuple ones (rotation_order token, quat).

    MEMOISED per (sim, rec_id, rests), because the curves are stateless
    readers: for the same arguments every build produces an equivalent set,
    and the instance-list rebuild asks for ~900 links at ~30 curves each. On
    gat 2 that was 70k calls across 180 frames, the largest single term in
    the rebuild. The cache hangs off the SIM so it dies with it - keying on
    `id(sim)` would let a freed sim's address be reused by a new one.

    A COPY is returned: callers pin properties on the dict they get back
    (`_notefield_link` forces `hidden`), and a shared dict would leak that
    into every other link of the same actor."""
    from analysis.games.notitg.sim.seg_read import curve_for

    cache = getattr(sim, '_link_timeline_cache', None)
    if cache is None:
        cache = {}
        try:
            sim._link_timeline_cache = cache
        except AttributeError:  # a slotted stand-in: build every time
            cache = None
    key = (rec_id, None if not rests else tuple(sorted(rests.items())))
    if cache is not None and key in cache:
        return dict(cache[key])

    merged = {**_LINK_RESTS, **(rests or {})}
    timelines = {prop: curve_for(sim, rec_id, prop, rest)
                 for prop, rest in merged.items()}
    for prop, rest in _TUPLE_LINK_RESTS.items():
        timelines[prop] = curve_for(sim, rec_id, prop, rest)
    if cache is not None:
        cache[key] = timelines
    return dict(timelines)


_EASE_LINEAR = 0

# How far apart two samples must be before the span between them counts as
# motion rather than float noise, in design pixels / degrees.
_FLAT_EPS = 1e-9

# How finely a CURVED span is subdivided when a sum is read back across it.
# The sum is read as straight lines between the times its parts change, and
# a curve is not straight, so it contributes a grid of its own.
_TRACE_DT = 1.0 / 60.0


def _add_part_times(times: set, exported, t0: float, t1: float) -> None:
    """Add one part's breakpoint times to `times`, subdividing any span it
    eases NON-LINEARLY so the chords across it follow the curve."""
    ts, _vals, durs, eases = exported
    for bt, dur, ease in zip(ts, durs, eases):
        if t0 < bt < t1:
            times.add(bt)
        if ease == _EASE_LINEAR or dur <= 0.0:
            continue
        steps = max(1, int(np.ceil(dur / _TRACE_DT)))
        for k in range(1, steps):
            traced = bt + k * dur / steps
            if t0 < traced < t1:
                times.add(traced)


class _SumTimeline:
    """Additive overlay: oscillator deltas riding on a recorded stream."""

    def __init__(self, timelines):
        self._timelines = tuple(timelines)

    def sample(self, t):
        return (sum(tl.sample(t)[0] for tl in self._timelines),)

    def breakpoints(self, t0: float, t1: float, index: int = 0):
        """`(ts, vals, durs, eases)` reproducing `sample(t)[0]` over
        `[t0, t1]` for a piecewise linear-ramp consumer, or None when a part
        cannot describe its own shape.

        BETWEEN two consecutive part breakpoints no part changes, so the sum
        is one straight line there and reading it back describes that line
        exactly. Which is what makes the sum exportable at all: two channels
        cannot be added breakpoint-wise, but they can be re-read on the union
        of their breakpoint times. A part that CURVES between its own
        breakpoints subdivides its span into that union, since a curve is
        the one thing a straight-line reading cannot carry.

        A part carrying a step is why a ramp's TARGET can need its own
        breakpoint: the consumer ramps toward `vals[i+1]`, which is not the
        value that must be served at that time. The two share the instant,
        and the consumer's bisect takes the later.

        Sampling the sum on a fixed grid instead - the exporter fallback -
        aliases whatever the deltas do between grid points.
        """
        times = self._union_times(t0, t1, index)
        if times is None:
            return None
        ts: list[float] = []
        vals: list[float] = []
        durs: list[float] = []

        def emit(t, value, dur):
            ts.append(float(t))
            vals.append(float(value))
            durs.append(float(dur))

        target = None
        for a, b in zip(times, times[1:]):
            head, tail = self._straight(a, b)
            if target is not None and abs(target - head) > _FLAT_EPS:
                emit(a, target, 0.0)
            moving = abs(tail - head) > _FLAT_EPS
            emit(a, head, b - a if moving else 0.0)
            target = tail if moving else None
        last = self.sample(times[-1])[0]
        if target is not None and abs(target - last) > _FLAT_EPS:
            emit(times[-1], target, 0.0)
        emit(times[-1], last, 0.0)
        return ts, vals, durs, [_EASE_LINEAR] * len(ts)

    def _straight(self, a: float, b: float) -> tuple:
        """`(value just after a, value just before b)` for the straight line
        the sum follows across `[a, b]`, read from two INTERIOR samples.

        The ends themselves are not sampled. A step lands ON a union time,
        and which side of it a float reads is not something the caller
        controls: a vibrate rolls its cell as `int((t - start) * hz)`, so
        `start + k / hz` comes back as cell k or cell k-1 depending on the
        last bit, and one wrong cell is a whole random offset. Both quarter
        points are unambiguously inside, and the line through them
        extrapolates to both ends."""
        lo = self.sample(a + (b - a) * 0.25)[0]
        hi = self.sample(a + (b - a) * 0.75)[0]
        if abs(hi - lo) <= _FLAT_EPS:
            level = 0.5 * (lo + hi)
            return level, level
        return 1.5 * lo - 0.5 * hi, 1.5 * hi - 0.5 * lo

    def _union_times(self, t0: float, t1: float, index: int):
        """Every time any part changes shape, plus the window's own ends -
        or None when a part cannot describe its own shape at all."""
        times = {float(t0), float(t1)}
        for part in self._timelines:
            export = getattr(part, 'breakpoints', None)
            if export is None:
                return None
            exported = export(t0, t1, index)
            if exported is None:
                return None
            _add_part_times(times, exported, t0, t1)
        return sorted(times)


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
    before the first beat). `flip_base_y` mirrors the leaf's vertical
    source axis: AFT samplers set basezoomy(-1) to compensate the
    engine's bottom-up GL captures, and our capture is already top-down,
    so the sign convention is translated (a chart leaving basezoomy at
    +1 to want the raw flipped texture still lands). The mirror is the
    INNERMOST factor (content-side), so canceling the sign alone is not
    enough: the anchor offset negates (valign 0.75 on a flipped sprite
    places the quad like valign 0.25 upright) and the vertical crop
    edges swap (`crop_at`) - the engine's cropbottom hides the flipped
    quad's screen-TOP content. Outer rotation/skew are untouched (the
    mirror never crosses them).

    `.at(t)` returns `(H, alpha)` - H the 3x3 row-vector homography
    mapping capture coords onto the design screen - or None when the
    instance is invisible: hidden anywhere in the chain, alpha ~0,
    degenerate scale, or tilted fully behind the eye."""

    def __init__(self, links, t0=None, flip_base_y=False):
        self._links = tuple(links)
        self._t0 = t0
        self._flip_base_y = flip_base_y

    def at(self, t):
        sampled = self.project_at(t)
        if sampled is None:
            return None
        H, alpha = sampled
        if self._culled(H, t):
            return None
        return H, alpha

    def project_at(self, t):
        """`at` minus the face-culling gate: the composed `(H, alpha)` or
        None on the visibility/degeneracy gates alone. The doc's cull-gate
        synthesis samples this - it needs the H a culled frame WOULD have
        drawn with, which `at` by design no longer returns."""
        if self._t0 is not None:
            t = max(float(t), self._t0)
        alpha = 1.0
        world = None
        fov = field_projection.FOV
        leaf = len(self._links) - 1
        for i, link in enumerate(self._links):
            if link['hidden'].sample(t)[0] >= 0.5:
                return None
            if link['awake'].sample(t)[0] < 0.5:
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

    def _culled(self, H, t) -> bool:
        """Whether face culling drops this draw at `t`: the leaf's SetCullMode
        against the projected winding.

        Winding is judged in ENGINE terms. The chart's `basezoomy(-1)` itself
        reverses winding - an AFT sampler is deliberately BACK-facing at rest,
        which is how it survives `cullmode('front')` - and `flip_base_y`
        translates that sign away, so a flipped leaf's engine winding is the
        NEGATED determinant of ours. The two-sided-card idiom is the consumer:
        front/back sprite pairs, the back at rotationx+180, both cull 'front',
        so the projected flip picks exactly one face per frame (gat 2's
        chicken finale froze upside-down when both drew)."""
        mode = self.cull_mode_at(t)
        if mode < 0.5:
            return False
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if self._flip_base_y:
            det = -det
        return det > 0.0 if mode >= 1.5 else det < 0.0

    def cull_mode_at(self, t) -> float:
        """The leaf's cull mode at `t`: 0 none, 1 back, 2 front. A link set
        built before the lane existed reads as none."""
        timeline = self._links[-1].get('cull') if self._links else None
        return timeline.sample(t)[0] if timeline is not None else 0.0

    def culled_at(self, H, t) -> bool:
        """Public face of the cull judgment for a caller that already holds
        the projected H (`project_at`) - the doc's gate synthesis."""
        return self._culled(H, t)

    def may_draw(self, t) -> bool:
        """True unless the chain's VISIBILITY gates already rule this
        instance out at `t` - hidden, asleep, or faded past `_MIN_ALPHA`
        anywhere along it.

        The cheap half of `at`: three samples per link and no geometry.
        `at` applies the same gates and then folds the transform, which can
        rule out more (a degenerate scale, a plane turned past the eye), so
        a True here is 'in play', not 'drawn'. For a caller that only needs
        to know whether an instance is in play - which capture scopes a
        frame must prepare for, say - that is the whole question, at a
        sixth of the cost."""
        if self._t0 is not None:
            t = max(float(t), self._t0)
        alpha = 1.0
        for link in self._links:
            if link['hidden'].sample(t)[0] >= 0.5:
                return False
            if link['awake'].sample(t)[0] < 0.5:
                return False
            alpha *= link['alpha'].sample(t)[0]
        return alpha >= _MIN_ALPHA

    def crop_at(self, t):
        """The instance's crop insets `(left, top, right, bottom)` as
        fractions of its texture, or None at rest (no crop - today's
        exact blit). Crop is read from the LEAF link only: SetCrop*
        insets the drawn quad of the actor that owns a texture, and only
        the leaf sprite has one (an ActorFrame's crop is a no-op). A
        flipped (aft) leaf swaps top/bottom: the crop names the engine
        quad's edges, and the source mirror puts cropbottom's hidden
        band at OUR source's top (the afthell half-screen bands)."""
        if self._t0 is not None:
            t = max(float(t), self._t0)
        link = self._links[-1]
        left, top, right, bottom = (link[prop].sample(t)[0]
                                    for prop in _CROP_PROPS)
        if self._flip_base_y:
            top, bottom = bottom, top
        crop = (left, top, right, bottom)
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
            if flip:
                # The source mirror is innermost, so the anchor offset
                # rides it: a flipped sprite's valign places the quad on
                # the opposite side of the position.
                ady = -ady

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
    sampler), 'fill' (an AFT-rig curtain quad blitted at its tree
    position), 'capture' (a chain-involved AFT node: snapshot the
    in-progress composite into the node's named slot at this tree
    position), or 'stage' (an isolating AFT node: its captured sprite's
    transform, folded into downstream consumers at sample time - never
    a draw of its own); `player` is the 1-based player whose capture the
    instance blits (0 for the rest). `aft_order` places an 'aft' sampler
    relative to its source AFT node in draw order: 'post' samplers show
    the frame's fresh capture (drawn after the node captured), 'pre'
    samplers show the previous frame's (their draw preceded this frame's
    capture). `aft_live` samples the source node's visibility (0.0 =
    hidden = the preserve-texture capture is frozen); `color` samples a
    'fill' curtain's diffuse rgb. A 'stage' instance's links are the
    captured SPRITE's chain (an aft sampler leaf, so it shares the
    source flip semantics)."""
    return {'name': name, 'kind': kind, 'player': player,
            'aft_order': aft_order, 'aft_live': aft_live, 'color': color,
            'transform': TransformChannel(links, t0=t0,
                                          flip_base_y=kind in ('aft',
                                                               'stage'))}


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


def playfield_mod_link(channels, number) -> dict | None:
    """Player `number`'s PLAYFIELD TRANSFORM MODS as one link, or None when
    the chart drives none of them.

    `x`/`y`/`z`/`rotation*`/`zoom*`/`skewx` move the whole notefield rather
    than each arrow (`mod_channels.PLAYFIELD_MODS`), so they belong on the
    field's own chain, innermost of it: the engine turns and scales the
    field about its own middle and then the chart's actors carry it around
    the screen, which is what an innermost link composes. Reusing the actor
    link keeps them on one transform path - out-of-plane rotation gets the
    chain's real perspective fold rather than a per-arrow approximation.

    The mod curves are already piecewise linear, so they republish as
    keyframes exactly (`EventTimeline` eases from the previous target),
    which is also what lets the drawable doc export them breakpoint for
    breakpoint instead of dense-sampling."""
    if channels is None:
        return None
    keyframes = {}
    for mod, (prop, unit) in mod_channels.PLAYFIELD_MODS.items():
        times, values = channels.breakpoints(mod, number - 1)
        if times:
            keyframes[prop] = _mod_keyframes(times, values, unit)
    return link_timelines(keyframes) if keyframes else None


def _mod_keyframes(times, values, unit) -> list:
    """One mod curve's `(times, values)` as `Keyframe`s in the property's
    own unit. Each point tweens toward the NEXT point's value over the gap
    between them; the last holds. Points sharing a time are the curve's
    vertical steps (an instant retarget) - the later one wins the sample,
    so emitting both is exact."""
    last = len(times) - 1
    return [Keyframe(t=times[i],
                     values=(values[min(i + 1, last)] * unit,),
                     duration=(times[i + 1] - times[i]) if i < last else 0.0,
                     easing=_EASE_LINEAR)
            for i in range(len(times))]


def player_instance(number, keyframes, osc_deltas=None, t0=None,
                    channels=None) -> dict:
    """A player-field instance drawn in place from its own link."""
    link = player_link(number, keyframes, osc_deltas)
    return instance(f'P{number}', 'player', number,
                    with_playfield_mods([link], channels, number), t0=t0)


def player_live_instance(sim, number, rec_id, osc_deltas=None, t0=None,
                         channels=None) -> dict:
    """LAZY player-field instance: LiveCurves over the live PlayerP{n} actor."""
    link = player_live_link(sim, number, rec_id, osc_deltas)
    return instance(f'P{number}', 'player', number,
                    with_playfield_mods([link], channels, number), t0=t0)


def screen_centered_link() -> dict:
    """A link that lands a full-screen capture 1:1 on the design screen.

    `TransformChannel` centres the content before folding the chain, so a
    chain that never positions it draws half a screen up and left. This is
    the link a consumer with no actor of its own composes against - the base
    field, which is otherwise an identity blit."""
    return link_timelines(None, rests={'x': _CENTER_X, 'y': _CENTER_Y})


def with_playfield_mods(links, channels, number) -> list:
    """`links` with player `number`'s playfield mod link appended, when the
    chart drives any. Innermost, so the mods act in the field's own space
    and everything above still carries the result (`playfield_mod_link`)."""
    mods = playfield_mod_link(channels, number)
    return [*links, mods] if mods is not None else list(links)


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
