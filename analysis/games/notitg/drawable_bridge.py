"""NotITG -> Drawable doc: the first game producer for the Drawable core.

The Drawable model (`.claude/plans/drawable-ir.md`) folds a game's per-frame
composite into a game-agnostic document (Seam A) that a Rust evaluator turns
into a flat op stream each frame (Seam B). This module bridges the NotITG
compiled chart onto that model:

- `build_doc(compiled, ...)` walks the chart's distinct AFT/capture slot names
  and mints one PERSISTENT drawable per slot (engine AFT preserve-texture
  semantics: a slot retains its content across frames), plus one DYNAMIC
  drawable that carries the screen's per-frame entry stream. It returns the
  finished `Evaluator` and the id maps naming both.
- `feed_frame(compiled, t, id_maps)` samples the SAME
  `NotitgFieldInstances` effect the renderer samples (`effect.at(ctx)` over a
  design `chart_rect` of (0, 0, 640, 480)) and translates each field entry into
  the FROZEN feed record layout (u32 stride 4, f32 stride 14). It returns the
  four flat SoA feed buffers `Evaluator.frame_with_feeds` ingests.

Translation rules (per the wave-2 spec section B3):

- An entry's screen transform is a Qt `QTransform`. With a design chart_rect of
  (0, 0, 640, 480) the design map is the identity, so the screen transform IS
  the design-space homography. AFFINE entries whose linear block is a pure
  rotate+axis-scale (no shear, no perspective) decompose EXACTLY into the feed's
  (x, y, sx, sy, rot) TRS lanes; sheared / projective entries do not fit those
  lanes (they await B1's full-link path) and are SKIPPED with a coverage count.
- Scope 'fill' -> `SRC_FILL` with the entry's rgb as tint.
- Scope 'capture' -> emits NOTHING on the feed: snapshot topology is static, so
  the `Snapshot` command lives in the doc (see `build_doc`).
- Scope 'screen' / 'screen_prev' (aft blits) -> `SRC_DRAWABLE` of the mapped
  slot drawable.
- Scope 'field' / 'field{N}' (proxy / player blits) -> `SRC_DRAWABLE` of the
  matching per-player field drawable.

No Qt import at module load - the bridge is importable headless. All
`QTransform` handling stays inside the functions (the effect itself pulls Qt in
only when sampled).
"""
from __future__ import annotations

import math

import numpy as np

# The design chart region: SM's fixed 640x480 screen. Sampling the effect over
# this region makes the design map the identity (kx=ky=1, ox=oy=0), so the
# entry's screen QTransform equals its design-space homography - no extra
# conjugation to undo before decomposing.
_DESIGN_RECT = (0.0, 0.0, 640.0, 480.0)
_SCREEN_W = 640.0
_SCREEN_H = 480.0

# The one dynamic drawable id (per-frame entry stream). Drawable 0 is always the
# screen root; the dynamic feed drawable is minted right after it.
_SCREEN_ID = 0

# Feed record layout (frozen; mirrors native/src/evaluate.rs FEED_*_STRIDE):
#   u32 stride 4: [source_kind, source_id, frame, flags]
#   f32 stride 14: [x, y, sx, sy, rot, opacity, r, g, b,
#                   crop_l, crop_t, crop_r, crop_b, z]
_FEED_U_STRIDE = 4
_FEED_F_STRIDE = 14
_FEED_FLAG_ADDITIVE = 1 << 0

# Orthogonality tolerance for the affine-representability check: the feed's
# linear block T(x,y).R(rot).S(sx,sy) has orthogonal columns by construction,
# so an entry fits IFF its two linear columns are orthogonal to this bound (a
# sheared entry fails it). Sub-visible in normalized column space.
_ORTHO_ATOL = 1e-4


class _Ctx:
    """The minimal effect-sampling context: `NotitgFieldInstances.at` reads
    only `t_now` and `chart_rect`."""

    __slots__ = ('t_now', 'chart_rect')

    def __init__(self, t, chart_rect):
        self.t_now = t
        self.chart_rect = chart_rect


def _slot_names(compiled) -> list[str]:
    """The distinct AFT/capture slot names an aft/capture entry can reference,
    in a stable order. An aft sampler's slot key is its source node (the freeze
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


def _current_instances(compiled) -> list:
    """The current field-instance list: `field_instances` is a provider
    callable (lazy topology) or a fixed sequence."""
    provider = compiled.get('field_instances')
    if provider is None:
        return []
    return list(provider() if callable(provider) else provider)


def build_doc(compiled, screen_w: float = _SCREEN_W, screen_h: float = _SCREEN_H):
    """Build the Drawable doc for a compiled NotITG chart.

    Returns `(evaluator, id_maps)` where `evaluator` is a finished
    `storyboard_native.Evaluator` and `id_maps` is a dict with:
      - `'screen'`   : the screen root drawable id (0)
      - `'dynamic'`  : the per-frame entry-stream (dynamic) drawable id
      - `'slots'`    : {slot name -> persistent drawable id} (AFT / capture)
      - `'fields'`   : {field scope -> field-source drawable id} (proxy/player)

    One PERSISTENT drawable per distinct AFT/capture slot name (a slot retains
    content across frames, the engine's preserve-texture semantics); one
    per-player field-source drawable per distinct field scope; one DYNAMIC
    drawable for the screen's per-frame entry stream. The screen root blits the
    dynamic drawable once, so the whole per-frame entry stream composes in
    order.
    """
    import storyboard_native as sn

    builder = sn.DocBuilder(float(screen_w), float(screen_h))
    # Drawable 0 is minted by the builder as the screen root.
    dynamic_id = builder.drawable(float(screen_w), float(screen_h),
                                  persistent=False, dynamic=True)

    field_ids: dict[str, int] = {}
    for scope in _field_scopes(compiled):
        field_ids[scope] = builder.drawable(float(screen_w), float(screen_h),
                                            persistent=False, dynamic=False)

    slot_ids: dict[str, int] = {}
    for name in _slot_names(compiled):
        slot_ids[name] = builder.drawable(float(screen_w), float(screen_h),
                                         persistent=True, dynamic=False)

    # The screen root composes the per-frame entry stream (the dynamic feed
    # drawable) once, in insertion order.
    builder.item(_SCREEN_ID, sn.SRC_DRAWABLE, dynamic_id)

    evaluator = builder.finish()
    id_maps = {'screen': _SCREEN_ID, 'dynamic': dynamic_id,
               'slots': slot_ids, 'fields': field_ids}
    return evaluator, id_maps


def _resolve_source(entry_scope, extra, id_maps):
    """The `(source_kind, source_id)` a translated entry blits, or None when
    the entry is not a feed item (a 'capture' entry -> static Snapshot in the
    doc, emitted nowhere on the feed).

    Returns a special `('fill', tint)` marker for a fill scope so the caller
    reads the tint from `extra`; else the drawable-source pair.
    """
    import storyboard_native as sn

    match entry_scope:
        case 'fill':
            tint = extra if isinstance(extra, tuple) else (1.0, 1.0, 1.0)
            return ('fill', tint)
        case 'capture':
            # A static Snapshot in the doc, never a feed item.
            return None
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


def _decompose_affine(qt):
    """Decompose a Qt `QTransform` into the feed's (x, y, sx, sy, rot_deg) TRS
    lanes, or None when the transform does not fit (projective, or a sheared
    linear block).

    The feed's mat3 is `translate(x, y) . rotate(rot) . scale(sx, sy)` (rust
    evaluate.rs), whose linear columns are `sx*(cos, sin)` and `sy*(-sin, cos)`
    - orthogonal by construction. The executor reads the mat3 from the Qt
    accessors as `col0 = (m11, m12)`, `col1 = (m21, m22)`, `translate =
    (dx, dy)`. So an affine fits IFF those two columns are orthogonal; the angle
    comes from col0 and sy carries col1's sign.
    """
    if not qt.isAffine():
        return None
    m11, m12 = qt.m11(), qt.m12()
    m21, m22 = qt.m21(), qt.m22()
    col0_n = math.hypot(m11, m12)
    col1_n = math.hypot(m21, m22)
    if col0_n < 1e-12 or col1_n < 1e-12:
        return None
    if abs(m11 * m21 + m12 * m22) > _ORTHO_ATOL * col0_n * col1_n:
        return None
    rot = math.atan2(m12, m11)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    sy = m21 * (-sin_r) + m22 * cos_r
    return qt.dx(), qt.dy(), col0_n, sy, math.degrees(rot)


def _entry_transform(entry_transform):
    """(x, y, sx, sy, rot_deg) for an entry's screen QTransform, or None when
    it is not affine-decomposable. `None` transform is the untouched identity
    (a centered original blit)."""
    if entry_transform is None:
        return (0.0, 0.0, 1.0, 1.0, 0.0)
    return _decompose_affine(entry_transform)


def feed_frame(compiled, t, id_maps):
    """Sample the field-instance effect at `t` and translate its entries into
    the frozen feed SoA buffers.

    Returns `(feed_ids, feed_item_counts, feed_u_bytes, feed_f_bytes,
    coverage)`. The first four feed straight into
    `Evaluator.frame_with_feeds(t, feed_ids, feed_item_counts, feed_u,
    feed_f)`; `coverage` is a dict `{'translated', 'skipped_projective',
    'total'}` reporting how many entries fit the TRS feed lanes.

    Entry order is preserved: the op stream's blit order matches the sampled
    entry order (SortSpan / z reordering already happened inside the effect's
    `at`).
    """
    from analysis.games.notitg.field_instances import NotitgFieldInstances

    provider = compiled.get('field_instances')
    effect = NotitgFieldInstances(provider, base_hidden=compiled.get('base_field_hidden'))
    frame = effect.at(_Ctx(float(t), _DESIGN_RECT))
    entries = frame.fields if frame is not None else ()

    dynamic_id = id_maps['dynamic']

    u_rows: list[tuple] = []
    f_rows: list[list] = []
    translated = 0
    skipped_projective = 0
    total = 0
    for entry in entries:
        transform, alpha, scope, extra = _unpack_entry(entry)
        source = _resolve_source(scope, extra, id_maps)
        if source is None:
            # A 'capture' entry (static Snapshot in the doc) or an
            # unresolved slot - not a feed item.
            continue
        total += 1
        trs = _entry_transform(transform)
        if trs is None:
            skipped_projective += 1
            continue
        x, y, sx, sy, rot = trs
        crop = _unpack_crop(entry)
        source_kind, source_id, tint, additive = _source_lanes(source)
        flags = _FEED_FLAG_ADDITIVE if additive else 0
        u_rows.append((source_kind, source_id, 0, flags))
        f_rows.append([x, y, sx, sy, rot, float(alpha),
                       tint[0], tint[1], tint[2],
                       crop[0], crop[1], crop[2], crop[3], 0.0])
        translated += 1

    n = len(u_rows)
    feed_u = np.array(u_rows, dtype=np.uint32).reshape(n, _FEED_U_STRIDE) \
        if n else np.zeros((0, _FEED_U_STRIDE), dtype=np.uint32)
    feed_f = np.array(f_rows, dtype=np.float32).reshape(n, _FEED_F_STRIDE) \
        if n else np.zeros((0, _FEED_F_STRIDE), dtype=np.float32)
    coverage = {'translated': translated,
                'skipped_projective': skipped_projective,
                'total': total}
    # One feed targets the single dynamic drawable; its item count is n.
    return ([dynamic_id], [n], feed_u.tobytes(), feed_f.tobytes(), coverage)


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
