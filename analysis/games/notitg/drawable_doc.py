"""NotITG static scene -> channel-backed DrawableDoc in engine tree order.

The Drawable model (`.claude/plans/drawable-ir.md`) folds a game's per-frame
composite into a game-agnostic document (Seam A) a Rust evaluator turns into a
flat op stream each frame (Seam B). This module compiles the STATIC part of a
NotITG chart - the field-instance topology (player fields, proxy re-renders, AFT
samplers, fills, capture snapshots) - into that document ONCE, in engine tree
order, so the static scene no longer rides per-frame feeds.

It is the tree-order sibling of `drawable_bridge` (which routes the same
field-instance stream through per-frame dynamic feeds): where the bridge re-emits
the whole entry stream every frame, this module STATICALLY places each instance
as an `item_link` chain whose per-property channels are exported once at build
time. The Rust evaluator then samples those channels itself - no Python in the
per-frame loop for the static part.

`build_static_doc(compiled)` walks the provider's instance list in order and,
per instance:

  - proxy / player field blits -> `SRC_DRAWABLE` of the per-player field
    drawable, transform = the instance's full link chain via `item_link`;
  - aft samplers -> `SRC_DRAWABLE` of the sampler's freeze-key slot drawable,
    same link chain, with the aft vertical flip on the leaf link;
  - fills -> `SRC_FILL` with the sampled diffuse rgb as tint channels;
  - captures -> a `Snapshot` command into the capture's named slot drawable,
    placed at the instance's tree position (no draw of its own);
  - a `z_group` run of instances is wrapped in a `SortSpan` so the evaluator
    stably re-sorts it by each item's sampled `z` (SetDrawByZPosition).

`export_channel` is THE shared primitive: it converts one of the Python timeline
objects an instance's links carry (an `EventTimeline`, or a lazy
`LiveCurve`/`SegCurve`) into the `(ts, vals, durs)` breakpoint triple
`DocBuilder.channel` ingests, exactly for `EventTimeline` and by dense sampling
for the lazy curves. It is documented for later extraction (other games' static
scenes will reuse it verbatim).

The parity harness (`compare_at` / `parity_report`) compares the evaluator's
BLIT stream against `NotitgFieldInstances.at` at N sample times - order, alpha,
and the design-space mat3 - and is the deliverable quality bar.

Scope note (native transform coverage): the Rust `compose_links` path models the
2D affine link chain only (x/y, scale, in-plane rotation, base-scale, anchor,
crop, flip) - NOT z-translation, rotation_x/y, quaternion spin, skew, or the fov
perspective the Python `TransformChannel` also folds. Instances exercising those
diverge from the Python homography BY DESIGN (they await the camera-area port);
the parity harness reports such divergences rather than hiding them.

No Qt import at module load - importable headless.
"""
from __future__ import annotations

import math

import numpy as np

from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.storyboard.sprite_sheet import frame_at_time

# The design chart region: SM's fixed 640x480 screen. Sampling the effect over
# this region makes the design map the identity (kx=ky=1, ox=oy=0), so an
# entry's screen QTransform equals its design-space homography - the same space
# the DrawableDoc's natural-640x480 link chains live in, so the two compare
# directly with no conjugation.
_DESIGN_RECT = (0.0, 0.0, 640.0, 480.0)
_SCREEN_W = 640.0
_SCREEN_H = 480.0

# Drawable 0 is always the screen root (minted by the builder).
_SCREEN_ID = 0

# The lazy-curve sampling cadence (LiveCurve / SegCurve have no exposed
# keyframes, so they are densely sampled). 1/30s: the interim approximation the
# spec calls out - an EventTimeline is exported exactly instead.
_DENSE_DT = 1.0 / 30.0

# Linear easing id (effects.easing): the ONLY easing the native channel's
# breakpoint ramp reproduces exactly. Other eases within a keyframe span are
# densified.
_EASE_LINEAR = 0

_REST_EPS = 1e-4

# Storyboard element kinds this wave emits as Image items. Sprites (a single
# cell) and frame-animated sheets both back an SRC_IMAGE draw; every other kind
# (shapes, text, groups-with-no-image, video, compound) is skipped with a
# per-kind count in the report. Backgrounds - the user-reported missing content
# - are BGCHANGES sprites, so they land here.
_IMAGE_KINDS = ('sprite', 'frames')

# The element property -> native `item` transform-lane mapping. Each pair is
# (element timeline prop, the item() id/rest kwarg stem); every lane is a scalar
# EventTimeline exported once through export_channel. `hidden` is handled apart
# (inverted into the item's visible gate), and the sheet frame is its own
# derived channel - neither appears here.
_ELEMENT_ITEM_LANES = (
    ('x', 'x'), ('y', 'y'),
    ('scale_x', 'sx'), ('scale_y', 'sy'),
    ('rotation', 'rot'),
    ('alpha', 'opacity'),
)


# --------------------------------------------------------------------------
# export_channel - the shared timeline -> breakpoint-triple primitive
# --------------------------------------------------------------------------

def export_channel(timeline, t0: float, t1: float, prop: int = 0):
    """Convert one Python timeline into the `(ts, vals, durs)` breakpoint
    triple `DocBuilder.channel` ingests, over the window `[t0, t1]`.

    The native channel (native/src/channels.rs) serves `rest` before the first
    breakpoint, then at each breakpoint either holds its value (dur <= 0) or
    ramps LINEARLY toward the next breakpoint's value over `dur`. This helper
    emits breakpoints reproducing the timeline's own playback under that model:

    - `EventTimeline`: EXACT. Its keyframes are translated directly - an instant
      (duration 0) becomes one hold breakpoint; a linearly-eased ramp becomes a
      (start-value, dur) breakpoint plus a (target, hold) breakpoint, matching
      EventTimeline's ease-from-previous playback; a non-linearly-eased ramp is
      densified across its own span at `_DENSE_DT` (the native ramp is linear,
      so a curved ease cannot be one breakpoint). Rest before the first keyframe
      is carried on the ChannelRef, so no pre-roll breakpoint is emitted.
    - anything else (`LiveCurve` / `SegCurve`, or any `.sample(t)` duck):
      dense-sampled at `_DENSE_DT` across `[t0, t1]` (the documented interim
      approximation - these lazy curves expose no keyframe structure).

    `prop` selects the value index within the timeline's sampled tuple (link
    timelines are scalar, index 0). Returns `(ts, vals, durs, rest)` - the four
    args of `DocBuilder.channel` - as plain Python lists + a float rest.

    Examples: called by `_channel_id` for every per-property link channel of a
    field instance, and by the fill-tint export.
    """
    if isinstance(timeline, EventTimeline):
        return _export_event_timeline(timeline, prop)
    return _export_dense(timeline, t0, t1, prop)


def _rest_value(timeline, prop: int) -> float:
    """The timeline's pre-first-keyframe rest value for `prop`. EventTimeline
    exposes it directly; a duck-typed curve is sampled far in the past."""
    rest = getattr(timeline, '_rest', None)
    if rest is not None:
        return float(rest[prop])
    return float(timeline.sample(-1.0e18)[prop])


def _export_event_timeline(timeline: EventTimeline, prop: int):
    """Exact breakpoints for an EventTimeline (see `export_channel`)."""
    keyframes = timeline._kf
    rest = _rest_value(timeline, prop)
    ts: list[float] = []
    vals: list[float] = []
    durs: list[float] = []

    def emit(bt: float, value: float, dur: float) -> None:
        # Collapse a redundant hold onto the previous breakpoint of equal value
        # (keeps the channel minimal; the native sampler is unaffected).
        if ts and durs[-1] <= 0.0 and abs(vals[-1] - value) <= _REST_EPS \
                and dur <= 0.0:
            return
        ts.append(bt)
        vals.append(value)
        durs.append(dur)

    prev = rest
    for idx, kf in enumerate(keyframes):
        target = float(kf.values[prop])
        if kf.duration <= 0.0:
            emit(kf.t, target, 0.0)
            prev = target
            continue
        start = float(kf.start[prop]) if kf.start is not None else prev
        if kf.easing == _EASE_LINEAR:
            emit(kf.t, start, kf.duration)
            emit(kf.t + kf.duration, target, 0.0)
        else:
            # A curved ease: the native ramp is linear, so densify the span.
            _densify_span(timeline, prop, kf.t, kf.t + kf.duration, ts, vals, durs)
            emit(kf.t + kf.duration, target, 0.0)
        prev = target

    return ts, vals, durs, rest


def _densify_span(timeline, prop, a: float, b: float, ts, vals, durs) -> None:
    """Append linear-ramp breakpoints tracing `timeline` across `[a, b)` at
    `_DENSE_DT`, so the piecewise-linear reconstruction follows the curve."""
    n = max(1, int(np.ceil((b - a) / _DENSE_DT)))
    step = (b - a) / n
    for k in range(n):
        bt = a + k * step
        ts.append(bt)
        vals.append(float(timeline.sample(bt)[prop]))
        durs.append(step)


def _export_dense(timeline, t0: float, t1: float, prop: int):
    """Dense-sample a duck-typed curve across `[t0, t1]` at `_DENSE_DT`
    (the interim approximation for LiveCurve / SegCurve)."""
    rest = _rest_value(timeline, prop)
    if t1 <= t0:
        return [], [], [], rest
    n = max(1, int(np.ceil((t1 - t0) / _DENSE_DT)))
    step = (t1 - t0) / n
    ts: list[float] = []
    vals: list[float] = []
    durs: list[float] = []
    for k in range(n + 1):
        bt = t0 + k * step
        ts.append(bt)
        vals.append(float(timeline.sample(bt)[prop]))
        durs.append(step if k < n else 0.0)
    return ts, vals, durs, rest


# --------------------------------------------------------------------------
# Storyboard elements - flatten, band, and (per kind) emit or skip
# --------------------------------------------------------------------------

# A group (ActorFrame) draws nothing itself; its transform composes onto its
# children. This wave does not yet fold the group transform into the leaves
# (the item() transform lanes carry each leaf's OWN timelines only), so a group
# is flattened away and its image leaves emit at their own placement - a
# documented interim (true group composition awaits the transform-fold wave).
_SKIP_GROUP = 'group'


class _FrameCurve:
    """A `.sample(t)`-duck curve tracing a sheet sprite's current frame index
    over time, so `export_channel` dense-samples it into the item's frame lane.

    Mirrors `render._sheet_frame`: a recorded `state_pin` sampler wins; else the
    sheet auto-animates through `sheet_states` on the element's own clock. A
    plain 1x1 sprite has no states and no pin, so this rests at frame 0."""

    __slots__ = ('_element', '_rest')

    def __init__(self, element):
        self._element = element
        self._rest = (0.0,)

    def sample(self, t: float) -> tuple:
        element = self._element
        if element.state_pin is not None:
            return (float(element.state_pin.sample(t)[0]),)
        return (float(frame_at_time(element.sheet_states, t - element.t_start)),)


def _flatten_elements(elements) -> list:
    """The element tree flattened to a leaf list in tree order, groups replaced
    by their children (see `_SKIP_GROUP`). Returns `(leaves, group_count)`."""
    leaves = []
    groups = 0
    stack = list(reversed(list(elements)))
    while stack:
        element = stack.pop()
        if element.kind == _SKIP_GROUP:
            groups += 1
            stack.extend(reversed(list(element.children)))
            continue
        leaves.append(element)
    return leaves, groups


def _band_elements(leaves):
    """Split flattened leaves into (below, above) bands by the sign of their
    band z, each band internally sorted by the renderer's `(z, z_index,
    t_start)` key - reproducing today's banding as the doc's starting point.
    Below-band (z < 0) draws before the field instances, above-band (z >= 0)
    after (true tree-order interleave replaces this in a later wave)."""
    ordered = sorted(leaves, key=lambda e: (e.z, e.z_index, e.t_start))
    below = [e for e in ordered if e.z < 0]
    above = [e for e in ordered if e.z >= 0]
    return below, above


# --------------------------------------------------------------------------
# build_static_doc - the tree-order static-scene compiler
# --------------------------------------------------------------------------

# The item_link parameters, in call order, paired with the link-timeline prop
# name they read. `zoom_*`/`rot` rename `scale_*`/`rotation`; crop lanes rename
# the crop_* edges; halign/valign/base_scale carry through. natural_w/h are
# constants (the 640x480 capture), passed as rests.
_LINK_PROP_ORDER = (
    ('x', 'x'), ('y', 'y'),
    ('zoom_x', 'scale_x'), ('zoom_y', 'scale_y'),
    ('rot', 'rotation'),
    ('skew_x', 'skew_x'), ('skew_y', 'skew_y'),
    ('base_scale_x', 'base_scale_x'), ('base_scale_y', 'base_scale_y'),
    ('halign', 'halign'), ('valign', 'valign'),
    ('hidden', 'hidden'), ('alpha', 'alpha'),
    ('crop_l', 'crop_left'), ('crop_t', 'crop_top'),
    ('crop_r', 'crop_right'), ('crop_b', 'crop_bottom'),
)


class _Ctx:
    """Minimal effect-sampling context: `NotitgFieldInstances.at` reads only
    `t_now` and `chart_rect`."""

    __slots__ = ('t_now', 'chart_rect')

    def __init__(self, t, chart_rect):
        self.t_now = t
        self.chart_rect = chart_rect


def _current_instances(compiled) -> list:
    """The current field-instance list: `field_instances` is a provider
    callable (lazy topology) or a fixed sequence."""
    provider = compiled.get('field_instances')
    if provider is None:
        return []
    return list(provider() if callable(provider) else provider)


def _aft_slot_key(inst) -> str:
    """The freeze/slot key an aft sampler blits (field_instances._extra): its
    isolated upstream capture_source, else its source node, else its own name."""
    return (inst.get('capture_source') or inst.get('aft_node') or inst['name'])


def _horizon(compiled, default: float = 600.0) -> float:
    """The channel export window end. Prefer an explicit compiled horizon;
    fall back to a generous default (channels hold their tail past it)."""
    for key in ('duration', 'horizon', 'song_length'):
        value = compiled.get(key)
        if isinstance(value, (int, float)) and value > 0.0:
            return float(value)
    return default


class _Builder:
    """Threads the DocBuilder through one static-doc build: mints the field /
    slot drawables, exports channels (de-duplicating identical timelines), and
    emits each instance's commands. Not reused across builds."""

    def __init__(self, compiled, screen_w, screen_h, builder=None):
        import storyboard_native as sn

        self._sn = sn
        self._compiled = compiled
        self._t0 = 0.0
        self._t1 = _horizon(compiled)
        # `builder` injection: a _RecordingBuilder lets the EXPENSIVE part of
        # a build (the channel exports - tens of seconds of pure Python) run
        # on a worker thread; the recorded ops replay onto a real DocBuilder
        # on the thread the unsendable Evaluator must live on (see
        # prepare_static_doc / assemble_static_doc).
        self._builder = builder if builder is not None \
            else sn.DocBuilder(float(screen_w), float(screen_h))
        self._screen = (float(screen_w), float(screen_h))
        self._field_ids: dict[str, int] = {}
        self._slot_ids: dict[str, int] = {}
        # Storyboard image sources: an image id per DISTINCT absolute asset path
        # (loading is the consumer's job). id_maps carries the {id -> path} map.
        self._image_ids: dict[str, int] = {}
        self._image_paths: dict[int, str] = {}
        self._image_grids: dict[int, tuple] = {}
        # Per-band emitted counts + per-kind skip counts, surfaced in the report.
        self._elem_below = 0
        self._elem_above = 0
        self._elem_skips: dict[str, int] = {}
        # Memoize channel ids by timeline object identity + prop, so a link's
        # shared rest timelines (a whole field of untouched props) collapse to
        # one channel each.
        self._chan_cache: dict[tuple[int, int], tuple[int, float]] = {}

    # -- drawable minting -------------------------------------------------

    def _new_drawable(self, persistent: bool) -> int:
        return self._builder.drawable(self._screen[0], self._screen[1],
                                      persistent=persistent, dynamic=False)

    def _field_drawable(self, scope: str) -> int:
        drawable = self._field_ids.get(scope)
        if drawable is None:
            drawable = self._new_drawable(persistent=False)
            self._field_ids[scope] = drawable
        return drawable

    def _slot_drawable(self, name: str) -> int:
        drawable = self._slot_ids.get(name)
        if drawable is None:
            drawable = self._new_drawable(persistent=True)
            self._slot_ids[name] = drawable
        return drawable

    # -- channel export ---------------------------------------------------

    def _channel(self, timeline, prop: int = 0) -> tuple[int, float]:
        """The (channel_id, rest) for a timeline+prop, exported once and
        memoized. id < 0 sentinel is never returned here - a real channel is
        always pushed (the rest still rides the ChannelRef)."""
        key = (id(timeline), prop)
        cached = self._chan_cache.get(key)
        if cached is not None:
            return cached
        ts, vals, durs, rest = export_channel(timeline, self._t0, self._t1, prop)
        chan_id = self._builder.channel(
            [float(v) for v in ts], [float(v) for v in vals],
            [float(v) for v in durs], float(rest))
        result = (chan_id, float(rest))
        self._chan_cache[key] = result
        return result

    def _link_kwargs(self, link, flip_base_y: bool) -> dict:
        """The `item_link` keyword args for one link dict: an (id, rest) channel
        per transform prop, natural size as constant rests, and the leaf flip."""
        kwargs: dict[str, object] = {}
        for param, prop in _LINK_PROP_ORDER:
            timeline = link.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{param}_id'] = chan_id
            kwargs[f'{param}_rest'] = rest
        kwargs['natural_w_rest'] = _SCREEN_W
        kwargs['natural_h_rest'] = _SCREEN_H
        kwargs['flip_base_y'] = flip_base_y
        return kwargs

    def _emit_links(self, target: int, inst) -> None:
        """Attach the instance's full leaf-link chain (root-first) to the item
        most recently pushed onto `target`. The leaf link carries the aft flip
        (field_compose sets TransformChannel.flip_base_y for aft/stage kinds)."""
        links = inst['transform']._links
        flip = getattr(inst['transform'], '_flip_base_y', False)
        leaf = len(links) - 1
        for i, link in enumerate(links):
            self._builder.item_link(
                target, **self._link_kwargs(link, flip and i == leaf))

    # -- instance emission ------------------------------------------------

    def _emit_blit(self, source_kind: int, source_id: int, inst,
                   additive: bool = False) -> None:
        z_id, z_rest, has_z = self._z_channel(inst)
        self._builder.item(_SCREEN_ID, source_kind, source_id,
                           additive=additive, z_id=z_id, z_rest=z_rest,
                           has_z=has_z)
        self._emit_links(_SCREEN_ID, inst)

    def _z_channel(self, inst):
        """(z_id, z_rest, has_z) for an instance's SortSpan sort key. The
        z_sort timeline is scalar; absent -> no z (pure insertion order).

        A fill's time-varying diffuse rgb (`inst['color']`) is NOT plumbed: the
        native `item` seeds the BLIT tint white and offers no per-channel tint
        setter, so a colored curtain composes white. The parity harness compares
        source / mat3 / alpha, not tint, so this is invisible there; a fill's
        real color awaits a tint-channel item API (documented limitation)."""
        z_sort = inst.get('z_sort')
        if inst.get('z_group') is None or z_sort is None:
            return -1, 0.0, False
        chan_id, rest = self._channel(z_sort)
        return chan_id, rest, True

    # -- entry point ------------------------------------------------------

    def run(self):
        """Emit the below-band storyboard elements, then the field-instance
        stream in tree order, then the above-band elements, and finish.

        Reproducing today's banding: below-band (z < 0) storyboard elements draw
        BEFORE the field instances, above-band (z >= 0) AFTER. Within each band
        the leaves are sorted by the renderer's `(z, z_index, t_start)` key. This
        is the starting point - a true tree-order interleave of elements and
        instances replaces it once the producers emit element tree positions."""
        below, above = self._banded_elements()
        self._elem_below = self._emit_element_band(below)

        instances = self._current_instances_ensured()
        self._emit_base_field(instances)
        i = 0
        n = len(instances)
        while i < n:
            inst = instances[i]
            group = inst.get('z_group')
            if group is not None:
                span_len = self._emit_z_run(instances, i, group)
                i += span_len if span_len else 1
                continue
            self._emit_instance(inst)
            i += 1

        self._elem_above = self._emit_element_band(above)

        evaluator = self._builder.finish()
        id_maps = {'screen': _SCREEN_ID, 'slots': dict(self._slot_ids),
                   'fields': dict(self._field_ids),
                   'images': dict(self._image_paths),
                   'image_grids': dict(self._image_grids)}
        return evaluator, id_maps

    def _banded_elements(self):
        """(below, above) storyboard-element bands from `compiled['tree']`,
        groups flattened away (their leaves banded by their own z). The group
        count is folded into the skip tally so the report accounts for it."""
        tree = self._compiled.get('tree') or ()
        leaves, groups = _flatten_elements(tree)
        if groups:
            self._elem_skips['group'] = self._elem_skips.get('group', 0) + groups
        return _band_elements(leaves)

    def _current_instances_ensured(self) -> list:
        instances = _current_instances(self._compiled)
        self._ensure_player_consumers(instances)
        return instances

    def _emit_base_field(self, instances) -> None:
        """The single-player base original: an identity blit of the primary
        'field' capture, drawn FIRST (mirrors NotitgFieldInstances._single_frame
        prepending `(None, 1.0, 'field')`). Gated visible by `1 - base_hidden`,
        so when the chart hides the real field the item drops (matching the
        base-hidden placeholder). Suppressed on the dual-player path, where the
        player instances ARE the originals and no base is prepended."""
        spec = self._compiled.get('player_fields')
        if spec is not None and getattr(spec, 'note_mods', None):
            return
        field = self._field_drawable('field')
        visible_id, visible_rest = self._visible_from_hidden(
            self._compiled.get('base_field_hidden'))
        self._builder.item(_SCREEN_ID, self._sn.SRC_DRAWABLE, field,
                           visible_id=visible_id, visible_rest=visible_rest)

    def _visible_from_hidden(self, hidden):
        """(visible_id, visible_rest) inverting a `base_field_hidden` timeline
        into the item's `visible` gate. Absent -> constant visible. A present
        timeline is exported inverted (hidden >= 0.5 -> visible < 0.5), so the
        native visibility gate drops the base while the chart hides the field."""
        if hidden is None:
            return -1, 1.0
        ts, vals, durs, rest = export_channel(hidden, self._t0, self._t1)
        inv_vals = [1.0 - v for v in vals]
        chan_id = self._builder.channel(
            [float(v) for v in ts], inv_vals,
            [float(v) for v in durs], 1.0 - float(rest))
        return chan_id, 1.0 - float(rest)

    def _ensure_player_consumers(self, instances) -> None:
        """Mirror NotitgFieldInstances.at: mint per-player consumers before
        scopes resolve, so a proxy of player N > 1 keys its own field{N}."""
        spec = self._compiled.get('player_fields')
        if spec is None:
            return
        spec.ensure({player for inst in instances
                     if inst.get('kind') in ('proxy', 'player')
                     and (player := inst.get('player') or 1) > 1})

    def _emit_z_run(self, instances, start: int, group) -> int:
        """Emit a SortSpan wrapping the maximal run of z_group==`group`
        instances starting at `start`; returns the run length. Only emitted
        (drawable-producing) members count toward the span length."""
        members = []
        i = start
        while i < len(instances) and instances[i].get('z_group') == group:
            members.append(instances[i])
            i += 1
        run_len = i - start
        # The SortSpan precedes its members and names how many commands follow,
        # so its length counts only members that actually emit a command (a
        # 'stage' emits none). captures/fills/blits each emit exactly one.
        span_len = sum(1 for inst in members if inst.get('kind') != 'stage')
        self._builder.sort_span(_SCREEN_ID, span_len)
        for inst in members:
            self._emit_instance(inst)
        return run_len

    def _emit_instance(self, inst) -> None:
        sn = self._sn
        kind = inst['kind']
        match kind:
            case 'stage':
                # An isolating AFT node: never a draw of its own (its transform
                # is folded into downstream consumers at sample time). Skipped -
                # the static doc's fold is the Snapshot/slot topology.
                return
            case 'capture':
                slot = self._slot_drawable(inst['name'])
                self._builder.snapshot(_SCREEN_ID, slot)
            case 'fill':
                self._emit_blit(sn.SRC_FILL, 0, inst)
            case 'aft':
                slot = self._slot_drawable(_aft_slot_key(inst))
                additive = self._aft_additive(inst)
                self._emit_blit(sn.SRC_DRAWABLE, slot, inst, additive=additive)
            case 'player' | 'proxy':
                scope = self._field_scope(inst)
                drawable = self._field_drawable(scope)
                self._emit_blit(sn.SRC_DRAWABLE, drawable, inst)
            case _:
                return

    def _aft_additive(self, inst) -> bool:
        blend = inst.get('blend_add')
        return blend is not None and blend.sample(self._t0)[0] >= 0.5

    def _field_scope(self, inst) -> str:
        """The field-capture scope a proxy/player blits, mirroring
        NotitgFieldInstances._scope EXACTLY: the per-player 'field{N}' only when
        player N > 1 has a minted consumer in the player_fields spec, else the
        primary 'field'. A proxy of a player with no consumer falls to 'field'
        (single-player charts proxying a nominal player 2 still draw player 1's
        capture - the effect's own rule)."""
        from analysis.games.notitg.field_instances import _player_scope
        player = inst.get('player') or 1
        if player > 1 and player in self._player_note_mods():
            return _player_scope(player)
        return 'field'

    def _player_note_mods(self) -> dict:
        spec = self._compiled.get('player_fields')
        return getattr(spec, 'note_mods', {}) if spec is not None else {}

    # -- storyboard element emission --------------------------------------

    def _emit_element_band(self, leaves) -> int:
        """Emit each leaf in `leaves` as an Image item (or count its skip);
        returns the number of items actually emitted."""
        emitted = 0
        for element in leaves:
            if self._emit_element(element):
                emitted += 1
        return emitted

    def _emit_element(self, element) -> bool:
        """Emit one leaf element as an SRC_IMAGE item, or count it as a per-kind
        skip. Returns True when an item was emitted. Only image-backed kinds
        (sprite / frames) with a resolvable asset path draw; everything else -
        shapes, text, video, compound, an image kind with no asset - is skipped
        and tallied by kind (an asset-less image kind counts as 'no_asset')."""
        if element.kind not in _IMAGE_KINDS:
            self._count_skip(element.kind)
            return False
        image_id = self._image_id(element)
        if image_id is None:
            self._count_skip('no_asset')
            return False
        self._sn_image_item(image_id, element)
        return True

    def _count_skip(self, kind: str) -> None:
        self._elem_skips[kind] = self._elem_skips.get(kind, 0) + 1

    def _image_id(self, element):
        """The image id for an element's asset (a 'frames' element uses its
        first frame path), minted once per distinct absolute path and recorded
        in the image table. A sheet sprite also records its (cols, rows) grid
        so the executor can crop the CURRENT CELL (the `frame` lane indexes
        it) instead of drawing the whole sheet. First grid wins per path (two
        elements sharing a sheet with different grids would conflict; not
        seen in practice). None when the element carries no asset."""
        path = element.asset or (element.frames[0] if element.frames else None)
        if not path:
            return None
        image_id = self._image_ids.get(path)
        if image_id is None:
            image_id = len(self._image_ids)
            self._image_ids[path] = image_id
            self._image_paths[image_id] = path
        cols = int(getattr(element, 'sheet_cols', 1) or 1)
        rows = int(getattr(element, 'sheet_rows', 1) or 1)
        if cols * rows > 1 and image_id not in self._image_grids:
            self._image_grids[image_id] = (cols, rows)
        return image_id

    def _sn_image_item(self, image_id: int, element) -> None:
        """Push one SRC_IMAGE item: the element's scalar transform timelines on
        the item's own lanes (export_channel each), the inverted `hidden` gate
        on `visible`, and the sheet-frame channel on `frame`. The anchor/origin
        natural-size offset is NOT folded (item lanes are the leaf's own x/y);
        that placement refinement rides the transform-fold wave."""
        kwargs = self._element_transform_kwargs(element)
        kwargs.update(self._element_frame_kwarg(element))
        kwargs.update(self._element_visible_kwarg(element))
        self._builder.item(_SCREEN_ID, self._sn.SRC_IMAGE, image_id,
                           additive=bool(element.additive), **kwargs)

    def _element_transform_kwargs(self, element) -> dict:
        kwargs: dict[str, object] = {}
        for prop, stem in _ELEMENT_ITEM_LANES:
            timeline = element.timelines.get(prop)
            if timeline is None:
                continue
            chan_id, rest = self._channel(timeline)
            kwargs[f'{stem}_id'] = chan_id
            kwargs[f'{stem}_rest'] = rest
        return kwargs

    def _element_frame_kwarg(self, element) -> dict:
        """A sheet sprite carries a frame lane tracing its current cell; a plain
        1x1 sprite rests at frame 0 (no lane)."""
        if element.sheet_cols * element.sheet_rows <= 1 and element.state_pin is None:
            return {}
        chan_id, rest = self._channel(_FrameCurve(element))
        return {'frame_id': chan_id, 'frame_rest': rest}

    def _element_visible_kwarg(self, element) -> dict:
        """The element's visible gate: its existence WINDOW [t_start, t_end)
        (the legacy walk draws an element only inside it - render.py's
        `child.t_start <= t < child.t_end`; without this every element drew
        for the whole chart at once) ANDed with the SM `hidden` bit
        inverted. Hidden flips are instant steps, so the merged gate is a
        step channel: the inverted-hidden value inside the window, 0
        outside it."""
        hidden = element.timelines.get('hidden')
        t0 = float(element.t_start)
        t1 = float(element.t_end)
        open_start = t0 <= self._t0 + 1e-9
        open_end = (not math.isfinite(t1)) or t1 >= self._t1 - 1e-9
        if hidden is None and open_start and open_end:
            return {}
        if hidden is None:
            ts, vals = [t0], [1.0]
            if not open_end:
                ts.append(t1)
                vals.append(0.0)
            chan_id = self._builder.channel(
                ts, vals, [0.0] * len(ts), 0.0)
            return {'visible_id': chan_id, 'visible_rest': 0.0}
        hts, hvals, _durs, hrest = export_channel(hidden, self._t0, self._t1)
        inv = [1.0 - float(v) for v in hvals]
        inv_rest = 1.0 - float(hrest)

        def gate_at(t):
            value = inv_rest
            for bt, bv in zip(hts, inv):
                if bt > t:
                    break
                value = bv
            return value

        ts, vals = [t0], [gate_at(t0)]
        for bt, bv in zip(hts, inv):
            if t0 < bt and (open_end or bt < t1):
                ts.append(float(bt))
                vals.append(bv)
        if not open_end:
            ts.append(t1)
            vals.append(0.0)
        chan_id = self._builder.channel(ts, vals, [0.0] * len(ts),
                                        0.0 if not open_start else inv_rest)
        return {'visible_id': chan_id,
                'visible_rest': 0.0 if not open_start else inv_rest}


def build_static_doc(compiled, screen_w: float = _SCREEN_W,
                     screen_h: float = _SCREEN_H):
    """Compile a NotITG chart's static field-instance scene into a
    channel-backed DrawableDoc, in engine tree order.

    Returns `(evaluator, id_maps, report)`:
      - `evaluator`  : a finished `storyboard_native.Evaluator`;
      - `id_maps`    : `{'screen', 'slots': {key -> id}, 'fields': {scope -> id},
                         'images': {image_id -> absolute path}}`;
      - `report`     : `{'instances', 'fields', 'slots', 'captures', 'fills',
                         'aft', 'proxy', 'z_groups', 'images',
                         'elements_below', 'elements_above', 'element_skips'}`.

    The screen root drawable (0) carries the whole static command list; the
    storyboard element tree (`compiled['tree']`) draws around the field-instance
    stream by z band (below-band elements first, then the instances, then
    above-band elements), reproducing today's banding. proxy/player/aft blits
    source per-player field / per-slot drawables (minted lazily as referenced),
    captures emit Snapshot commands, fills emit SRC_FILL items, image elements
    emit SRC_IMAGE items (paths collected in `id_maps['images']`), and z_group
    runs are wrapped in SortSpans.
    """
    b = _Builder(compiled, screen_w, screen_h)
    instances = _current_instances(compiled)
    report = _report(instances)
    evaluator, id_maps = b.run()
    report['fields'] = len(id_maps['fields'])
    report['slots'] = len(id_maps['slots'])
    report['images'] = len(id_maps['images'])
    report['elements_below'] = b._elem_below
    report['elements_above'] = b._elem_above
    report['element_skips'] = dict(b._elem_skips)
    return evaluator, id_maps, report


class _RecordingBuilder:
    """Records DocBuilder calls as (method, args, kwargs) ops, minting the
    SAME sequential ids a real DocBuilder would (channels count from 0,
    drawables from 1 - the screen root is 0). The point: `_Builder`'s
    emission is tens of seconds of pure-Python channel export, but the PyO3
    Evaluator is unsendable - so the emission runs against this recorder on
    a WORKER thread, and `assemble_static_doc` replays the cheap FFI calls
    where the evaluator must live (the render thread)."""

    def __init__(self):
        self.ops: list[tuple] = []
        self._channels = 0
        self._drawables = 1

    def channel(self, *args, **kwargs) -> int:
        self.ops.append(('channel', args, kwargs))
        self._channels += 1
        return self._channels - 1

    def drawable(self, *args, **kwargs) -> int:
        self.ops.append(('drawable', args, kwargs))
        self._drawables += 1
        return self._drawables - 1

    def item(self, *args, **kwargs) -> None:
        self.ops.append(('item', args, kwargs))

    def item_link(self, *args, **kwargs) -> None:
        self.ops.append(('item_link', args, kwargs))

    def snapshot(self, *args, **kwargs) -> None:
        self.ops.append(('snapshot', args, kwargs))

    def sort_span(self, *args, **kwargs) -> None:
        self.ops.append(('sort_span', args, kwargs))

    def finish(self):
        return None


def prepare_static_doc(compiled, screen_w: float = _SCREEN_W,
                       screen_h: float = _SCREEN_H):
    """The worker-safe half of `build_static_doc`: run the full emission
    against a `_RecordingBuilder` and return `(ops, id_maps, report)` - all
    plain Python data, no PyO3 objects, safe to build on any thread. Feed
    the ops to `assemble_static_doc` on the consuming thread."""
    recorder = _RecordingBuilder()
    b = _Builder(compiled, screen_w, screen_h, builder=recorder)
    instances = _current_instances(compiled)
    report = _report(instances)
    _evaluator, id_maps = b.run()
    report['fields'] = len(id_maps['fields'])
    report['slots'] = len(id_maps['slots'])
    report['images'] = len(id_maps['images'])
    report['elements_below'] = b._elem_below
    report['elements_above'] = b._elem_above
    report['element_skips'] = dict(b._elem_skips)
    return recorder.ops, id_maps, report


def assemble_static_doc(ops, screen_w: float = _SCREEN_W,
                        screen_h: float = _SCREEN_H):
    """Replay recorded ops onto a real DocBuilder and finish it - the cheap
    FFI half, run on the thread the Evaluator lives on."""
    import storyboard_native as sn

    builder = sn.DocBuilder(float(screen_w), float(screen_h))
    for method, args, kwargs in ops:
        getattr(builder, method)(*args, **kwargs)
    return builder.finish()


def _report(instances) -> dict:
    kinds = [inst.get('kind') for inst in instances]
    groups = {inst.get('z_group') for inst in instances
              if inst.get('z_group') is not None}
    return {
        'instances': len(instances),
        'captures': kinds.count('capture'),
        'fills': kinds.count('fill'),
        'aft': kinds.count('aft'),
        'proxy': kinds.count('proxy') + kinds.count('player'),
        'z_groups': len(groups),
        'fields': 0,
        'slots': 0,
        'images': 0,
        'elements_below': 0,
        'elements_above': 0,
        'element_skips': {},
    }


# --------------------------------------------------------------------------
# Parity harness - evaluator BLIT stream vs NotitgFieldInstances.at
# --------------------------------------------------------------------------

# Op / source kinds (mirror native/src/evaluate.rs); imported lazily so this
# module stays importable without the extension for the pure helpers above.
_OP_BLIT = 1

# The source kind storyboard element blits carry (mirrors evaluate.rs). Field
# instances only ever blit fills (SRC_FILL) and drawables (SRC_DRAWABLE), so an
# image blit is unambiguously a storyboard element - the parity harness drops
# them to recover the field-instance subsequence.
_SRC_IMAGE = 0


def _blit_stream(evaluator, t):
    """The evaluator's BLIT ops at `t` as `(source_kind, source_id, mat3, alpha)`
    tuples, in draw order. mat3 is the design-space 3x3 (row-major, column-vector
    convention - the BLIT record layout)."""
    u_stride = evaluator.u_stride
    f_stride = evaluator.f_stride
    u_bytes, f_bytes, _uf, n = evaluator.frame(float(t))
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, u_stride)
    f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, f_stride)
    out = []
    for i in range(n):
        if u[i, 0] != _OP_BLIT:
            continue
        mat = f[i, 0:9].reshape(3, 3).astype(np.float64)
        alpha = float(f[i, 9])
        out.append((int(u[i, 1]), int(u[i, 2]), mat, alpha))
    return out


def _field_blit_subsequence(evaluator, t):
    """The field-instance SUBSEQUENCE of the blit stream at `t`: the full blit
    stream with storyboard element (SRC_IMAGE) blits dropped. With elements
    banded around the instances, this is exactly the stream the instance-only
    doc produced, in the same order - the parity harness compares against this
    so element inclusion never perturbs field-instance parity."""
    return [blit for blit in _blit_stream(evaluator, t) if blit[0] != _SRC_IMAGE]


def _mat3_from_qtransform(qt):
    """The design-space 3x3 (column-vector convention, matching the BLIT
    record) for a field entry's screen QTransform, or the identity when the
    entry transform is None (a centered original blit)."""
    if qt is None:
        return np.eye(3)
    # Qt row-vector storage: p' = p @ [[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]].
    # The BLIT record is column-vector (M @ p) with the same content transposed.
    return np.array([[qt.m11(), qt.m21(), qt.m31()],
                     [qt.m12(), qt.m22(), qt.m32()],
                     [qt.m13(), qt.m23(), qt.m33()]], dtype=np.float64)


def _mat3_rel_err(a, b) -> float:
    """Relative max-abs error between two 3x3 mats, each first normalized so
    its bottom-right entry is 1 (projective scale-invariance)."""
    a = a / a[2, 2] if abs(a[2, 2]) > 1e-12 else a
    b = b / b[2, 2] if abs(b[2, 2]) > 1e-12 else b
    scale = max(1.0, float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def compare_at(evaluator, id_maps, effect, t,
               alpha_atol: float = 1e-3, mat_rtol: float = 1e-2) -> dict:
    """Compare the evaluator's BLIT stream at `t` against the field effect's
    entries, in order. Returns a per-time diff dict:

      `{'t', 'ok', 'n_blit', 'n_entry', 'order_ok', 'diffs': [...],
        'max_alpha_err', 'max_mat_err'}`

    where `diffs` lists `(index, reason, detail)` for each mismatch. An entry
    whose source does not resolve to a doc drawable (a 'capture' snapshot, an
    unresolved slot) is not a BLIT and is skipped on BOTH sides so the streams
    stay aligned. mat3 comparison is projective-relative (both normalized), the
    same space on both sides (design 640x480 via `_DESIGN_RECT`).

    The evaluator side is the field-instance SUBSEQUENCE (storyboard element
    image blits dropped): with elements banded around the instances, that
    subsequence is exactly the instance-only doc's blit stream, so this parity
    is unchanged by element inclusion. `n_blit` counts the field blits only.
    """
    frame = effect.at(_Ctx(float(t), _DESIGN_RECT))
    entries = frame.fields if frame is not None else ()
    expected = _expected_blits(entries, id_maps)
    got = _field_blit_subsequence(evaluator, t)

    diffs = []
    max_alpha_err = 0.0
    max_mat_err = 0.0
    order_ok = len(expected) == len(got)
    for i in range(min(len(expected), len(got))):
        (e_kind, e_id, e_mat, e_alpha) = expected[i]
        (g_kind, g_id, g_mat, g_alpha) = got[i]
        if (e_kind, e_id) != (g_kind, g_id):
            diffs.append((i, 'source', ((e_kind, e_id), (g_kind, g_id))))
            order_ok = False
            continue
        alpha_err = abs(e_alpha - g_alpha)
        mat_err = _mat3_rel_err(g_mat, e_mat)
        max_alpha_err = max(max_alpha_err, alpha_err)
        max_mat_err = max(max_mat_err, mat_err)
        if alpha_err > alpha_atol:
            diffs.append((i, 'alpha', (e_alpha, g_alpha)))
        if mat_err > mat_rtol:
            diffs.append((i, 'mat3', mat_err))
    for i in range(min(len(expected), len(got)), max(len(expected), len(got))):
        diffs.append((i, 'missing', ('entry' if i < len(expected) else 'blit')))

    return {'t': float(t), 'ok': not diffs and order_ok,
            'n_blit': len(got), 'n_entry': len(expected),
            'order_ok': order_ok, 'diffs': diffs,
            'max_alpha_err': max_alpha_err, 'max_mat_err': max_mat_err}


def _expected_blits(entries, id_maps):
    """The field entries that become BLITs, in order, as
    `(source_kind, source_id, mat3, alpha)` - mirroring `_emit_instance`'s
    source resolution so the two streams line up."""
    import storyboard_native as sn

    out = []
    for entry in entries:
        transform = entry[0]
        alpha = float(entry[1]) if len(entry) > 1 else 1.0
        scope = entry[2] if len(entry) > 2 else 'field'
        extra = entry[3] if len(entry) > 3 else None
        # The native emit drops opacity < 1/255 (the base-hidden placeholder,
        # a faded copy); skip those here so the streams stay aligned.
        if alpha < 1.0 / 255.0:
            continue
        source = _resolve_entry_source(sn, scope, extra, id_maps)
        if source is None:
            continue
        out.append((source[0], source[1], _mat3_from_qtransform(transform), alpha))
    return out


def _resolve_entry_source(sn, scope, extra, id_maps):
    """(source_kind, source_id) an entry blits, or None when it is not a BLIT
    (a 'capture' snapshot, or an unresolved slot / field)."""
    match scope:
        case 'fill':
            return (sn.SRC_FILL, 0)
        case 'capture':
            return None
        case 'screen' | 'screen_prev':
            key = extra[0] if isinstance(extra, tuple) and extra else None
            slot = id_maps['slots'].get(key)
            return None if slot is None else (sn.SRC_DRAWABLE, slot)
        case _:
            field = id_maps['fields'].get(scope)
            return None if field is None else (sn.SRC_DRAWABLE, field)


def parity_report(evaluator, id_maps, effect, sample_times,
                  alpha_atol: float = 1e-3, mat_rtol: float = 1e-2) -> dict:
    """`compare_at` over `sample_times`; returns `{'times': [...],
    'all_ok', 'max_alpha_err', 'max_mat_err', 'n_fail'}`."""
    times = [compare_at(evaluator, id_maps, effect, t, alpha_atol, mat_rtol)
             for t in sample_times]
    return {
        'times': times,
        'all_ok': all(r['ok'] for r in times),
        'max_alpha_err': max((r['max_alpha_err'] for r in times), default=0.0),
        'max_mat_err': max((r['max_mat_err'] for r in times), default=0.0),
        'n_fail': sum(0 if r['ok'] else 1 for r in times),
    }


def format_parity_report(report) -> str:
    """A one-line-per-sample-time human summary of a `parity_report`."""
    lines = [
        f"parity: {'OK' if report['all_ok'] else 'FAIL'} "
        f"({report['n_fail']}/{len(report['times'])} times failing) "
        f"max_alpha_err={report['max_alpha_err']:.2e} "
        f"max_mat_err={report['max_mat_err']:.2e}"
    ]
    for r in report['times']:
        detail = '' if r['ok'] else f"  diffs={r['diffs'][:4]}"
        lines.append(
            f"  t={r['t']:8.3f}  blits={r['n_blit']:>3} entries={r['n_entry']:>3} "
            f"order={'ok' if r['order_ok'] else 'BAD'} "
            f"a_err={r['max_alpha_err']:.2e} m_err={r['max_mat_err']:.2e}{detail}")
    return '\n'.join(lines)
