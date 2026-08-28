"""NotITG -> Drawable doc: the first game producer for the Drawable core.

The Drawable model (`.claude/plans/drawable-ir.md`) folds a game's per-frame
composite into a game-agnostic document (Seam A) that a Rust evaluator turns
into a flat op stream each frame (Seam B). This module bridges the NotITG
compiled chart onto that model:

- `build_doc(compiled, ...)` walks the chart's distinct AFT/capture slot names
  and mints one PERSISTENT drawable per slot (engine AFT preserve-texture
  semantics: a slot retains its content across frames), plus one DYNAMIC
  drawable PER INTER-CAPTURE SEGMENT of the screen's per-frame entry stream.
  It splits the screen's command list at the chart's capture positions - the
  entries before the first capture feed the first dynamic drawable, then a
  `Snapshot` command copies the in-progress composite into the capture's slot,
  then the next dynamic drawable carries the following entries, and so on. This
  is the AFT capture's at-position semantics: a snapshot freezes the composite
  as of its tree position, so a later aft sampler that blits the slot reads
  PRE-curtain content, not the finished frame. It returns the finished
  `Evaluator` and the id maps naming everything (including the ordered segment
  drawables and the capture order the doc was built for).
- `feed_frame(compiled, t, id_maps)` samples the SAME `NotitgFieldInstances`
  effect the renderer samples (`effect.at(ctx)` over a design `chart_rect` of
  (0, 0, 640, 480)) and routes each field entry to its inter-capture segment's
  feed, translating it into the FROZEN feed v3 record layout (u32 stride 7,
  f32 stride 22). It returns the four flat SoA feed buffers
  `Evaluator.frame_with_feeds` ingests.

Translation rules (per the wave-3 spec section C1 - feed v2):

- An entry's screen transform is a Qt `QTransform` - a full projective
  homography with a design chart_rect of (0, 0, 640, 480) (where the design
  map is the identity, so the screen transform IS the design-space
  homography). It crosses to the feed as a mat3 in the record's column-vector
  layout, written to the BLIT record VERBATIM - homographies included (the
  executor divides). No affine decomposition, no projective skip: EVERY entry
  crosses. Coverage is therefore `{translated, total}` with translated == total
  (an entry only fails to translate when its source cannot be resolved to a
  slot/field, which is a topology gap, not a transform one).
- Scope 'fill' -> `SRC_FILL` with the entry's rgb as tint.
- Scope 'capture' -> emits NOTHING on the feed: it is a segment boundary, and
  the `Snapshot` command lives in the doc (see `build_doc`). Encountering one
  advances the feed to the next segment.
- Scope 'screen' / 'screen_prev' (aft blits) -> `SRC_DRAWABLE` of the mapped
  slot drawable.
- Scope 'field' / 'field{N}' (proxy / player blits) -> `SRC_DRAWABLE` of the
  matching per-player field drawable.

The QTransform -> mat3 mapping (verified against the RasterExecutor's read
`QTransform(m[0], m[3], m[1], m[4], m[2], m[5])`): the record mat3 is the Qt
matrix TRANSPOSED - record `m[i][j] = qt.m(j+1)(i+1)`, so the projective row
(lanes 6/7/8) carries `qt.m13()/m23()/m33()` and the translation lands in
lanes 2/5 (`qt.dx()/dy()`). See `_qt_to_mat3`.

Lazy topology (signature-rebuild): the compiled chart's capture set grows as
the chart plays (proxy/AFT binds fire during playback). The doc's segment /
snapshot structure is minted from the capture positions the provider exposes
AT BUILD TIME, so it must be rebuilt when that set changes - mirroring
`_LiveFieldInstances`' signature-rebuild pattern. `topology_signature(compiled)`
is the cheap hashable check; `feed_frame` returns whether the doc is stale, and
`build_doc`-holding callers (the pipeline) rebuild on a signature change.

No Qt import at module load - the bridge is importable headless. All
`QTransform` handling stays inside the functions (the effect itself pulls Qt in
only when sampled).
"""
from __future__ import annotations

import numpy as np

# The design chart region: SM's fixed 640x480 screen. Sampling the effect over
# this region makes the design map the identity (kx=ky=1, ox=oy=0), so the
# entry's screen QTransform equals its design-space homography - no extra
# conjugation to undo.
_DESIGN_RECT = (0.0, 0.0, 640.0, 480.0)
_SCREEN_W = 640.0
_SCREEN_H = 480.0

# The screen root drawable id (always 0).
_SCREEN_ID = 0

# Feed v2 record layout (frozen; mirrors native/src/evaluate.rs FEED_*_STRIDE):
#   u32 stride 7: [source_kind, source_id, frame, flags, shader+1,
#                  x_offset, x_count] (the bridge stamps no note
#                  shaders; lanes 4..6 stay 0)
#   f32 stride 22: [m00, m01, m02, m10, m11, m12, m20, m21, m22,
#                   opacity, r, g, b, crop_l, crop_t, crop_r, crop_b, z,
#                   model_z, rot_x, rot_y, rot_z]
# `z` is the local SORT key; lanes 18..22 are the item's 3D model, gated by
# the HAS_MODEL flag. The bridge feeds flat quads, so it always writes 0
# there. The mat3 lanes 0..9 are in the BLIT record's column-vector
# convention and are written to the record verbatim.
_FEED_U_STRIDE = 7
_FEED_F_STRIDE = 22
_FEED_FLAG_ADDITIVE = 1 << 0

# The identity mat3 in the record's column-vector layout (an untouched original
# blit: no transform).
_IDENTITY_MAT3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


class _Ctx:
    """The minimal effect-sampling context: `NotitgFieldInstances.at` reads
    only `t_now` and `chart_rect`."""

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


def _capture_names(compiled) -> list[str]:
    """The capture slot names an entry snapshots into, in tree (instance-list)
    order - one segment boundary per capture. A capture's name is its own
    node name (`inst['name']`)."""
    return [inst['name'] for inst in _current_instances(compiled)
            if inst.get('kind') == 'capture']


def _slot_names(compiled) -> list[str]:
    """The distinct AFT/capture slot names an aft/capture entry references, in
    a stable order. An aft sampler's slot key is its source node (the freeze
    identity - the engine preserves per node), so the slots are exactly the
    distinct `aft_node` / `capture_source` / capture-name keys the current
    instance set carries. Player/proxy field scopes are NOT here (they map to
    the per-player field drawables)."""
    instances = _current_instances(compiled)
    names: dict[str, None] = {}
    for inst in instances:
        kind = inst.get('kind')
        if kind == 'capture':
            names.setdefault(inst['name'], None)
        elif kind == 'aft':
            key = (inst.get('capture_source') or inst.get('aft_node')
                   or inst['name'])
            names.setdefault(key, None)
    return list(names)


def _field_scopes(compiled) -> list[str]:
    """The distinct field-capture scopes proxy/player entries blit from: the
    primary 'field' plus any per-player 'field{N}'. Kept in a stable order so
    the drawable ids are deterministic."""
    from analysis.games.notitg.field_instances import _player_scope
    instances = _current_instances(compiled)
    scopes: dict[str, None] = {'field': None}
    for inst in instances:
        if inst.get('kind') in ('proxy', 'player'):
            player = inst.get('player') or 1
            scopes.setdefault(_player_scope(player), None)
    return list(scopes)


def topology_signature(compiled):
    """A cheap hashable signature of the doc's static structure: the ordered
    capture names (segment boundaries) plus the slot- and field-scope sets. An
    unchanged signature means `build_doc` would mint the same drawable
    topology, so a cached evaluator stays valid (only transforms/alphas change,
    and those ride the per-frame feed). A change means the caller must rebuild
    the doc (mirrors `_LiveFieldInstances`' signature-rebuild)."""
    return (tuple(_capture_names(compiled)),
            tuple(_slot_names(compiled)),
            tuple(_field_scopes(compiled)))


def build_doc(compiled, screen_w: float = _SCREEN_W, screen_h: float = _SCREEN_H):
    """Build the Drawable doc for a compiled NotITG chart.

    Returns `(evaluator, id_maps)` where `evaluator` is a finished
    `storyboard_native.Evaluator` and `id_maps` is a dict with:
      - `'screen'`    : the screen root drawable id (0)
      - `'segments'`  : the inter-capture dynamic drawable ids, in order
                        (len == captures + 1)
      - `'captures'`  : the capture slot names, in the order their Snapshots
                        sit between segments
      - `'slots'`     : {slot name -> persistent drawable id} (AFT / capture)
      - `'fields'`    : {field scope -> field-source drawable id} (proxy/player)
      - `'signature'` : the topology signature this doc was built for

    One PERSISTENT drawable per distinct AFT/capture slot name (a slot retains
    content across frames, the engine's preserve-texture semantics); one
    per-player field-source drawable per distinct field scope; and one DYNAMIC
    drawable per inter-capture SEGMENT of the screen's entry stream. The screen
    root composes segment 0, snapshots into the first capture slot, composes
    segment 1, snapshots into the second, and so on - so a slot freezes the
    composite as of its tree position (pre-curtain content).
    """
    import storyboard_native as sn

    builder = sn.DocBuilder(float(screen_w), float(screen_h))

    captures = _capture_names(compiled)
    # One dynamic drawable per inter-capture segment (captures + 1).
    segment_ids: list[int] = [
        builder.drawable(float(screen_w), float(screen_h),
                         persistent=False, dynamic=True)
        for _ in range(len(captures) + 1)
    ]

    field_ids: dict[str, int] = {}
    for scope in _field_scopes(compiled):
        field_ids[scope] = builder.drawable(float(screen_w), float(screen_h),
                                            persistent=False, dynamic=False)

    slot_ids: dict[str, int] = {}
    for name in _slot_names(compiled):
        slot_ids[name] = builder.drawable(float(screen_w), float(screen_h),
                                         persistent=True, dynamic=False)

    # The screen root composes each segment's per-frame feed in order, with a
    # Snapshot into the capture's slot between consecutive segments - the AFT
    # capture's at-position freeze.
    for i, seg in enumerate(segment_ids):
        builder.item(_SCREEN_ID, sn.SRC_DRAWABLE, seg)
        if i < len(captures):
            slot = slot_ids.get(captures[i])
            if slot is not None:
                builder.snapshot(_SCREEN_ID, slot)

    evaluator = builder.finish()
    id_maps = {'screen': _SCREEN_ID, 'segments': segment_ids,
               'captures': captures, 'slots': slot_ids, 'fields': field_ids,
               'signature': topology_signature(compiled)}
    return evaluator, id_maps


def _resolve_source(entry_scope, extra, id_maps):
    """The `(source_kind, source_id)` a translated entry blits, or None when
    the entry is not a feed item.

    Returns a special `('fill', tint)` marker for a fill scope so the caller
    reads the tint from `extra`, `('capture', None)` for a capture segment
    boundary, else the drawable-source pair.
    """
    import storyboard_native as sn

    match entry_scope:
        case 'fill':
            tint = extra if isinstance(extra, tuple) else (1.0, 1.0, 1.0)
            return ('fill', tint)
        case 'capture':
            # A segment boundary: the Snapshot is static in the doc, and this
            # entry advances the feed to the next segment (never a feed item).
            return ('capture', None)
        case 'screen' | 'screen_prev':
            # An aft sampler's freeze key names its slot drawable: the first
            # extra element (see field_instances._extra). Absent -> the
            # whole-screen read is out of a static topology's reach; skip.
            key = extra[0] if isinstance(extra, tuple) and extra else None
            slot = id_maps['slots'].get(key)
            return None if slot is None else (sn.SRC_DRAWABLE, slot)
        case _:
            # A proxy / player field blit: the scope ('field' / 'field{N}')
            # names the field-source drawable.
            field = id_maps['fields'].get(entry_scope)
            return None if field is None else (sn.SRC_DRAWABLE, field)


def _qt_to_mat3(qt):
    """The record's column-vector mat3 (lanes 0..9) for an entry's screen
    `QTransform`, or the identity for a `None` transform (a centered original
    blit).

    The record's mat3 is the Qt matrix TRANSPOSED: the executor reads it as
    `QTransform(m[0], m[3], m[1], m[4], m[2], m[5])` (an affine block plus the
    projective row in lanes 6/7/8), so `m[i][j] = qt.m(j+1)(i+1)`. This makes
    a full homography cross verbatim - the executor already divides by the
    projective row."""
    if qt is None:
        return _IDENTITY_MAT3
    return (qt.m11(), qt.m21(), qt.m31(),
            qt.m12(), qt.m22(), qt.m32(),
            qt.m13(), qt.m23(), qt.m33())


def feed_frame(compiled, t, id_maps):
    """Sample the field-instance effect at `t` and route its entries into the
    per-segment frozen feed v2 SoA buffers.

    Returns `(feed_ids, feed_item_counts, feed_u_bytes, feed_f_bytes,
    coverage)`. `feed_ids` are the segment dynamic drawable ids that carry
    entries, `feed_item_counts` their per-segment item counts, and the two
    buffers concatenate the items in that segment order. `coverage` is a dict
    `{'translated', 'total', 'stale'}`: `translated` == `total` under the mat3
    feed (every entry crosses; the only drop is an unresolved source, a
    topology gap), and `stale` is True when the chart's capture set has grown
    past the doc `id_maps` was built for (the caller should rebuild the doc).

    Entry order is preserved within each segment: the op stream's blit order
    matches the sampled entry order (SortSpan / z reordering already happened
    inside the effect's `at`).
    """
    from analysis.games.notitg.field_instances import NotitgFieldInstances

    provider = compiled.get('field_instances')
    effect = NotitgFieldInstances(provider, base_hidden=compiled.get('base_field_hidden'))
    frame = effect.at(_Ctx(float(t), _DESIGN_RECT))
    entries = frame.fields if frame is not None else ()

    segments = id_maps['segments']
    # Per-segment row accumulators (one list pair per dynamic drawable).
    seg_u: list[list[tuple]] = [[] for _ in segments]
    seg_f: list[list[list]] = [[] for _ in segments]
    seg_idx = 0
    last_seg = len(segments) - 1

    translated = 0
    total = 0
    for entry in entries:
        transform, alpha, scope, extra = _unpack_entry(entry)
        source = _resolve_source(scope, extra, id_maps)
        if source is None:
            # An unresolved slot/field (a topology gap) - not a feed item.
            continue
        if source[0] == 'capture':
            # Segment boundary: advance to the next segment's feed (the
            # Snapshot is a static doc command). Clamp at the last segment so
            # a stale capture set (more captures than the doc was built for)
            # keeps routing into the final segment rather than overrunning.
            if seg_idx < last_seg:
                seg_idx += 1
            continue
        total += 1
        mat = _qt_to_mat3(transform)
        crop = _unpack_crop(entry)
        source_kind, source_id, tint, additive = _source_lanes(source)
        flags = _FEED_FLAG_ADDITIVE if additive else 0
        seg_u[seg_idx].append((source_kind, source_id, 0, flags,
                               0, 0, 0))
        seg_f[seg_idx].append([*mat, float(alpha),
                               tint[0], tint[1], tint[2],
                               crop[0], crop[1], crop[2], crop[3], 0.0,
                               0.0, 0.0, 0.0, 0.0])
        translated += 1

    feed_ids, counts, u_all, f_all = _pack_segments(segments, seg_u, seg_f)
    stale = tuple(id_maps.get('signature') or ()) != topology_signature(compiled)
    coverage = {'translated': translated, 'total': total, 'stale': stale}
    return (feed_ids, counts, u_all.tobytes(), f_all.tobytes(), coverage)


def _pack_segments(segments, seg_u, seg_f):
    """Flatten the per-segment row lists into the parallel feed buffers. Every
    segment is emitted (in order) so the segment ids stay aligned with the doc;
    an empty segment contributes a zero count. Returns
    `(feed_ids, counts, feed_u, feed_f)`."""
    counts = [len(rows) for rows in seg_u]
    u_rows = [row for rows in seg_u for row in rows]
    f_rows = [row for rows in seg_f for row in rows]
    n = len(u_rows)
    feed_u = np.array(u_rows, dtype=np.uint32).reshape(n, _FEED_U_STRIDE) \
        if n else np.zeros((0, _FEED_U_STRIDE), dtype=np.uint32)
    feed_f = np.array(f_rows, dtype=np.float32).reshape(n, _FEED_F_STRIDE) \
        if n else np.zeros((0, _FEED_F_STRIDE), dtype=np.float32)
    return list(segments), counts, feed_u, feed_f


def _source_lanes(source):
    """(source_kind, source_id, tint, additive) for a resolved source.

    A `('fill', tint)` marker becomes `SRC_FILL` with that tint; a drawable
    source pair keeps a white tint (its color rides its own capture)."""
    import storyboard_native as sn

    if source[0] == 'fill':
        return sn.SRC_FILL, 0, source[1], False
    return source[0], source[1], (1.0, 1.0, 1.0), False


def _unpack_entry(entry):
    """(transform, alpha, scope, extra) from a field entry tuple. The scope /
    extra tail is optional (older `(transform, alpha)` / `(transform, alpha,
    scope)` shapes default to the primary field, no extra)."""
    transform = entry[0]
    alpha = entry[1] if len(entry) > 1 else 1.0
    scope = entry[2] if len(entry) > 2 else 'field'
    extra = entry[3] if len(entry) > 3 else None
    return transform, alpha, scope, extra


def _unpack_crop(entry):
    """The entry's (crop_l, crop_t, crop_r, crop_b) fractions, or zeros at
    rest. Crop rides the 5th tuple element (None at rest)."""
    crop = entry[4] if len(entry) > 4 else None
    if crop is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
