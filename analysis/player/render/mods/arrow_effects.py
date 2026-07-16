"""Per-note mod math: a vectorized port of OpenITG's ArrowEffects.cpp.

# What this is

The host renderer supplies, per visible note, a column index and a
`y_offset` (signed pixels from the receptor along the scroll axis, before
mods; positive = not yet arrived). This module implements the mods that
*consume* that y_offset and produce per-note position/rotation/alpha/zoom
contributions. It does NOT compute y_offset itself (speed/accel mods:
boost/brake/wave/boomerang/expand live in the host's y_offset pipeline).

Every mod is an independent, summed pure function of
(percent, column, y_offset, t_now, beat_now), exactly as in
ArrowEffects.cpp. `note_offsets` sums them all; `receptor_offsets` is the
same evaluated at y_offset = 0 (receptors don't scroll, but drunk/tornado
still displace them and confusion still rotates them).

# Coordinate + determinism substitutions (deviations from the engine)

- ARROW_SIZE = 64.0, SCREEN_HEIGHT = 480.0. OpenITG reads SCREEN_HEIGHT
  from the theme (ScreenDimensions.h); 480 is StepMania's virtual-res
  default and the value the ArrowEffects constants were tuned against.
  Exposed as args so a caller can match a different field height.

- Column x-offsets: OpenITG reads pStyle->m_ColumnInfo[...].fXOffset. For
  the standard evenly-spaced field that is (col - (N-1)/2) * ARROW_SIZE,
  which is what we compute. tornado/flip/invert use these offsets.

- WALL CLOCK -> SONG TIME. OpenITG's periodic mods sample
  RageTimer::GetTimeSinceStartFast() (wall clock since program start):
  tipsy, drunk, and blink. A replay needs determinism and scrub-exactness,
  so we substitute t_now (song seconds). This is the one behavioral
  divergence from the engine: in-engine these mods drift with real time
  independent of the song; here they are phase-locked to song time. Every
  such site is marked WALLCLOCK below.

- 2D rendering. We render a 2D playfield, so the engine's 3D outputs are
  reprojected:
    * GetZPos (bumpy) -> a per-note zoom multiplier. In 3D, +z moves a
      note toward the camera, which reads as larger; we map the z push to
      a scale about 1.0. Documented in `bumpy_zoom`. (The alternative,
      dropping bumpy, loses a mod the pilot chart uses 12x.)
    * GetRotationX (roll) and GetRotationY (twirl) are rotations about the
      horizontal / vertical axes - a 2D sprite can't tilt out of plane, so
      these are DEFERRED (return no contribution) and flagged in
      `note_offsets`'s docstring. GetRotationZ (dizzy/confusion) is an
      in-plane spin and IS implemented.

# NotITG extensions vs OpenITG

- movex / movey (per-column playfield translation): 100% = one ARROW_SIZE
  (64 px), verified against the NotITG modifiers docs. Not in OpenITG
  ArrowEffects; added as `movex_x` / `movey_y`. Per-column numbered
  variants (movex0..movexN, drunk0.., etc.) are handled by the channel
  layer passing a per-column percent array; every mod function here
  already takes per-note percents, so a numbered variant is just a
  percent array that is zero off its column. See `column_percents`.

- confusion: NotITG's song-beat-accumulating spin (percent is radians*100
  per the docs). OpenITG has no separate confusion; its dizzy is the
  note-relative spin. Both implemented (`rotation_z`).

# PORT BOUNDARY

Pure numpy over per-note arrays; no Qt, no engine, no globals. rand_seed
is fixed and unused by the implemented mods (the one RNG mod, random-speed,
lives in the host y_offset pipeline) but kept in the signature as the port
seam for future random mods.

# Examples of formula provenance (ArrowEffects.cpp line refs)

    drunk        GetXPos      :239
    tornado      GetXPos      :204-236
    flip         GetXPos      :240-253
    invert       GetXPos      :254-294
    beat         GetXPos      :296-339
    tipsy        GetYPos      :174-176
    dizzy        GetRotationZ :364-378
    bumpy        GetZPos      :522-531
    stealth/hidden/sudden/blink  ArrowGetPercentVisible :441-484
    mini/tiny (zoom)  GetCenterLine/ArrowGetReverseShiftAndScale :144, :389
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARROW_SIZE = 64.0
SCREEN_HEIGHT = 480.0
CENTER_LINE_Y = 160.0
FADE_DIST_Y = 40.0
PI = np.pi


def column_offsets(keycount: int, arrow_size: float = ARROW_SIZE) -> np.ndarray:
    """Pixel x-offset of each column from field center, standard
    evenly-spaced field: (col - (N-1)/2) * arrow_size. Matches
    pStyle->m_ColumnInfo[...].fXOffset for a plain N-key style."""
    cols = np.arange(keycount, dtype=np.float64)
    return (cols - (keycount - 1) / 2.0) * arrow_size


def column_percents(percent, cols: np.ndarray, keycount: int,
                    per_column: dict | None = None) -> np.ndarray:
    """Broadcast a mod percent to a per-note array, folding in NotITG
    numbered per-column variants.

    `percent` is the global value (drunk); `per_column` maps a column
    index to that column's own value (drunk0..). A per-column entry
    overrides the global for notes in that column; other notes keep the
    global. Returns one value per note (aligned with `cols`)."""
    out = np.full(cols.shape, float(percent), dtype=np.float64)
    if per_column:
        col_val = np.full(keycount, float(percent), dtype=np.float64)
        for c, v in per_column.items():
            if 0 <= c < keycount:
                col_val[c] = float(v)
        out = col_val[cols]
    return out


def _scale(x, l1, h1, l2, h2):
    return (x - l1) * (h2 - l2) / (h1 - l1) + l2


def _quantize(f, interval):
    return np.floor((f + interval / 2.0) / interval).astype(np.float64) * interval


def drunk_x(percent, cols, y_offset, t_now, keycount, arrow_size=ARROW_SIZE):
    """GetXPos drunk (ArrowEffects.cpp:239). WALLCLOCK -> t_now."""
    col_idx = cols.astype(np.float64)
    phase = t_now + col_idx * 0.2 + y_offset * 10.0 / SCREEN_HEIGHT
    return percent * (np.cos(phase) * arrow_size * 0.5)


def tornado_x(percent, cols, y_offset, keycount, arrow_size=ARROW_SIZE):
    """GetXPos tornado (ArrowEffects.cpp:204-236). Per note, the column's
    real x is mapped to [-1, 1] within a window of +/- iTornadoWidth
    columns (2 in wide fields >4 cols, else 3), an arccos gives a phase,
    y_offset advances it, and the cosine maps back to a windowed x."""
    xoffsets = column_offsets(keycount, arrow_size)
    wide = keycount > 4
    width = 2 if wide else 3

    col_i = cols.astype(np.int64)
    start = np.clip(col_i - width, 0, keycount - 1)
    end = np.clip(col_i + width, 0, keycount - 1)

    min_x = np.empty(cols.shape, dtype=np.float64)
    max_x = np.empty(cols.shape, dtype=np.float64)
    for i in range(cols.shape[0]):
        window = xoffsets[start[i]:end[i] + 1]
        min_x[i] = window.min()
        max_x[i] = window.max()

    real = xoffsets[col_i]
    span = np.where(max_x == min_x, 1.0, max_x - min_x)
    between = np.clip(_scale(real, min_x, min_x + span, -1.0, 1.0), -1.0, 1.0)
    rads = np.arccos(between) + y_offset * 6.0 / SCREEN_HEIGHT
    adjusted = _scale(np.cos(rads), -1.0, 1.0, min_x, max_x)
    return (adjusted - real) * percent


def _mirror_column_shift(percent, cols, keycount, permutation, arrow_size):
    xoffsets = column_offsets(keycount, arrow_size)
    new_cols = permutation[cols]
    return (xoffsets[new_cols] - xoffsets[cols]) * percent


def flip_permutation(keycount: int) -> np.ndarray:
    """GetXPos flip (ArrowEffects.cpp:240-253): col -> (N-1) - col."""
    return (keycount - 1) - np.arange(keycount)


def invert_permutation(keycount: int) -> np.ndarray:
    """GetXPos invert (ArrowEffects.cpp:254-294): mirror within each half
    of a single-side field (iNumColsPerSide = keycount). Ported column by
    column from the C integer logic; the middle branch maps a column to
    itself (the C SCALE degenerates to identity there)."""
    per_side = keycount
    left_of_mid = (per_side - 1) // 2
    right_of_mid = (per_side + 1) // 2
    out = np.empty(keycount, dtype=np.int64)
    for col in range(keycount):
        on_side = col % per_side
        if on_side <= left_of_mid:
            first, last = 0, left_of_mid
        elif on_side >= right_of_mid:
            first, last = right_of_mid, per_side - 1
        else:
            first = last = on_side
        new_on_side = on_side if first == last \
            else int(round(_scale(on_side, first, last, last, first)))
        out[col] = new_on_side
    return out


def flip_x(percent, cols, keycount, arrow_size=ARROW_SIZE):
    """GetXPos flip: mirror the whole field, move by x-offset difference."""
    return _mirror_column_shift(percent, cols, keycount,
                                flip_permutation(keycount), arrow_size)


def invert_x(percent, cols, keycount, arrow_size=ARROW_SIZE):
    """GetXPos invert: mirror within each half of the field."""
    return _mirror_column_shift(percent, cols, keycount,
                                invert_permutation(keycount), arrow_size)


def beat_x(percent, y_offset, beat_now, bps):
    """GetXPos beat (ArrowEffects.cpp:296-339): a periodic sideways shove
    keyed to the song beat, sped up on fast songs. `bps` is beats/sec
    (m_fCurBPS). Returns the per-note shift."""
    accel_time, total_time = 0.2, 0.5
    bpm = bps * 60.0
    div = max(1.0, np.trunc(bpm / 150.0))
    accel_time /= div
    total_time /= div

    beat = beat_now + accel_time
    beat /= div
    even = (int(beat) % 2) != 0

    if beat < 0:
        return np.zeros_like(y_offset)
    beat -= np.trunc(beat)
    beat += 1.0
    beat -= np.trunc(beat)
    if beat >= total_time:
        return np.zeros_like(y_offset)

    if beat < accel_time:
        amount = _scale(beat, 0.0, accel_time, 0.0, 1.0)
        amount *= amount
    else:
        amount = _scale(beat, accel_time, total_time, 1.0, 0.0)
        amount = 1.0 - (1.0 - amount) * (1.0 - amount)
    if even:
        amount *= -1.0

    shift = 20.0 * amount * np.sin(y_offset / 15.0 + PI / 2.0)
    return percent * shift


def movex_x(percent, arrow_size=ARROW_SIZE):
    """NotITG movex: 100% = one arrow width along x. `percent` is a
    per-note array (per-column variants handled upstream)."""
    return percent * arrow_size


def tipsy_y(percent, cols, t_now, arrow_size=ARROW_SIZE):
    """GetYPos tipsy (ArrowEffects.cpp:174-176). WALLCLOCK -> t_now.
    Columns bob along the scroll axis."""
    col_idx = cols.astype(np.float64)
    return percent * (np.cos(t_now * 1.2 + col_idx * 1.8) * arrow_size * 0.4)


def movey_y(percent, arrow_size=ARROW_SIZE):
    """NotITG movey: 100% = one arrow width along y."""
    return percent * arrow_size


def bumpy_zoom(percent, y_offset):
    """GetZPos bumpy (ArrowEffects.cpp:527-528) reprojected to 2D zoom.

    In 3D, fZPos += percent * 40 * sin(yoffset/16). +z is toward the
    camera => larger. We map the z push (range +/- 40*percent px) to a
    scale about 1.0 by SCREEN_HEIGHT: zoom = 1 + z / SCREEN_HEIGHT. This
    is a proxy (true perspective would divide by focal length); it keeps
    the bob visible without a 3D pipeline. Returns a per-note zoom mult."""
    z = percent * 40.0 * np.sin(y_offset / 16.0)
    return 1.0 + z / SCREEN_HEIGHT


def dizzy_rotation(percent, note_beat, beat_now):
    """GetRotationZ dizzy (ArrowEffects.cpp:364-378): spin proportional to
    beats-until-step, wrapped to a full turn, in degrees. `note_beat` is
    the note's own beat, `beat_now` the current song beat."""
    rot = (note_beat - beat_now) * percent
    rot = np.mod(rot, 2.0 * PI)
    return rot * 180.0 / PI


def confusion_rotation(percent, beat_now):
    """NotITG confusion: whole-field spin accumulating with the song beat.
    Docs: percent is radians * 100, so radians = percent * beat_now.
    Returned in degrees, one scalar broadcast to every note."""
    return percent * beat_now * 180.0 / PI


def _center_line(mini_percent):
    zoom = 1.0 - mini_percent * 0.5
    return CENTER_LINE_Y / zoom


def _hidden_sudden(hidden, sudden):
    return hidden * sudden


def percent_visible(percents, cols, y_pos):
    """ArrowGetPercentVisible (ArrowEffects.cpp:441-484). `y_pos` is the
    post-tipsy y position (host supplies; = y_offset when tipsy is off).
    Returns per-note visibility in [0, 1]. WALLCLOCK for blink -> handled
    by the caller passing t_now into the blink term via `percents`."""
    hidden = percents.get('hidden', 0.0)
    sudden = percents.get('sudden', 0.0)
    stealth = percents.get('stealth', 0.0)
    hidden_off = percents.get('hiddenoffset', 0.0)
    sudden_off = percents.get('suddenoffset', 0.0)
    blink_adjust = percents.get('_blink_adjust', 0.0)
    mini = percents.get('mini', 0.0)

    hs = _hidden_sudden(hidden, sudden)
    center = _center_line(mini)
    hidden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -1.0, -1.25) + center * hidden_off
    hidden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 0.0, -0.25) + center * hidden_off
    sudden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -0.0, 0.25) + center * sudden_off
    sudden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 1.0, 1.25) + center * sudden_off

    adjust = np.zeros(cols.shape, dtype=np.float64)
    if hidden != 0.0:
        ha = np.clip(_scale(y_pos, hidden_start, hidden_end, 0.0, -1.0), -1.0, 0.0)
        adjust = adjust + hidden * ha
    if sudden != 0.0:
        sa = np.clip(_scale(y_pos, sudden_start, sudden_end, -1.0, 0.0), -1.0, 0.0)
        adjust = adjust + sudden * sa
    if stealth != 0.0:
        adjust = adjust - stealth
    adjust = adjust + blink_adjust

    visible = np.clip(1.0 + adjust, 0.0, 1.0)
    return np.where(y_pos < 0.0, 1.0, visible)


def blink_adjust(percent, t_now):
    """APPEARANCE_BLINK term (ArrowEffects.cpp:470-475). WALLCLOCK ->
    t_now. Quantized sine flicker, returned as the additive visible
    adjustment (<= 0) already scaled by percent."""
    if percent == 0.0:
        return 0.0
    f = np.sin(t_now * 10.0)
    f = _quantize(f, 0.3333)
    return percent * _scale(f, 0, 1, -1.0, 0.0)


def alpha_from_visible(visible):
    """GetAlpha (ArrowEffects.cpp:486-494): a hard 0/1 cutoff at 0.5. We
    return the raw visibility as a float multiplier for smooth 2D
    compositing; call `np.where(visible > 0.5, 1.0, 0.0)` to match the
    engine's hard cut exactly."""
    return visible


def zoom_from_mini(mini_percent):
    """GetZoom / mini (ArrowEffects.cpp:389): 100% mini = half size, 200%
    = zero. NotITG tiny shares EFFECT_MINI. Returns a scalar zoom."""
    return 1.0 - mini_percent * 0.5


@dataclass(frozen=True)
class NoteOffsets:
    """Summed per-note contributions. dx/dy in pixels, rotation in
    degrees, alpha/zoom are multipliers (arrays aligned with `cols`)."""
    dx: np.ndarray
    dy: np.ndarray
    rotation_deg: np.ndarray
    alpha_mult: np.ndarray
    zoom: np.ndarray


def _get(p, name):
    v = p.get(name, 0.0)
    return v


def _per_note(p, name, cols, keycount):
    """A mod percent as a per-note array, folding numbered per-column
    variants (`<name>0`.. keys) via `column_percents`."""
    per_col = {}
    for c in range(keycount):
        key = f'{name}{c}'
        if key in p:
            per_col[c] = p[key]
    return column_percents(p.get(name, 0.0), cols, keycount, per_col or None)


def _active(percents, name, keycount):
    """A mod is active if its global percent is nonzero or any numbered
    per-column variant (`<name>0`..) is present."""
    return bool(percents.get(name, 0.0)) or \
        any(f'{name}{c}' in percents for c in range(keycount))


def _dx(percents, cols, y_offset, t_now, beat_now, keycount, bps, arrow_size):
    dx = np.zeros(cols.shape[0], dtype=np.float64)
    if _active(percents, 'drunk', keycount):
        dx += drunk_x(_per_note(percents, 'drunk', cols, keycount),
                      cols, y_offset, t_now, keycount, arrow_size)
    if _get(percents, 'tornado'):
        dx += tornado_x(_get(percents, 'tornado'), cols, y_offset, keycount, arrow_size)
    if _get(percents, 'flip'):
        dx += flip_x(_get(percents, 'flip'), cols, keycount, arrow_size)
    if _get(percents, 'invert'):
        dx += invert_x(_get(percents, 'invert'), cols, keycount, arrow_size)
    if _active(percents, 'beat', keycount):
        dx += beat_x(_per_note(percents, 'beat', cols, keycount),
                     y_offset, beat_now, bps)
    if _active(percents, 'movex', keycount):
        dx += movex_x(_per_note(percents, 'movex', cols, keycount), arrow_size)
    return dx


def _dy(percents, cols, t_now, keycount, arrow_size):
    dy = np.zeros(cols.shape[0], dtype=np.float64)
    if _active(percents, 'tipsy', keycount):
        dy += tipsy_y(_per_note(percents, 'tipsy', cols, keycount),
                      cols, t_now, arrow_size)
    if _active(percents, 'movey', keycount):
        dy += movey_y(_per_note(percents, 'movey', cols, keycount), arrow_size)
    return dy


def _rotation(percents, note_beats, beat_now, n):
    rot = np.zeros(n, dtype=np.float64)
    if _get(percents, 'dizzy'):
        rot += dizzy_rotation(_get(percents, 'dizzy'), note_beats, beat_now)
    if _get(percents, 'confusion'):
        rot += confusion_rotation(_get(percents, 'confusion'), beat_now)
    return rot


def _zoom(percents, y_offset, n):
    zoom = np.full(n, zoom_from_mini(_get(percents, 'mini')), dtype=np.float64)
    if _get(percents, 'bumpy'):
        zoom = zoom * bumpy_zoom(_get(percents, 'bumpy'), y_offset)
    return zoom


def _alpha(percents, cols, y_pos, t_now):
    vis = dict(percents)
    vis['_blink_adjust'] = blink_adjust(_get(percents, 'blink'), t_now)
    return alpha_from_visible(percent_visible(vis, cols, y_pos))


def note_offsets(percents: dict, cols: np.ndarray, y_offset: np.ndarray,
                 t_now: float, beat_now: float, keycount: int,
                 note_beats: np.ndarray | None = None,
                 bps: float = 2.0, arrow_size: float = ARROW_SIZE,
                 rand_seed: int = 0) -> NoteOffsets:
    """Sum every implemented note-position mod into a `NoteOffsets`.

    `percents` maps mod name -> value (fraction); numbered per-column
    variants (`drunk0`, `movex1`, ...) are picked up automatically.
    `cols` is an int array of column indices; `y_offset` the pre-mod
    signed scroll offset per note; `note_beats` the per-note beat (for
    dizzy; falls back to beat_now if omitted).

    DEFERRED (2D limitation, documented in the module header): roll
    (RotationX) and twirl (RotationY) are out-of-plane tilts a 2D sprite
    can't express, so they contribute nothing here."""
    cols = np.asarray(cols)
    y_offset = np.asarray(y_offset, dtype=np.float64)
    n = cols.shape[0]
    if note_beats is None:
        note_beats = np.full(n, beat_now, dtype=np.float64)
    else:
        note_beats = np.asarray(note_beats, dtype=np.float64)

    dx = _dx(percents, cols, y_offset, t_now, beat_now, keycount, bps, arrow_size)
    dy = _dy(percents, cols, t_now, keycount, arrow_size)
    rotation = _rotation(percents, note_beats, beat_now, n)
    zoom = _zoom(percents, y_offset, n)
    alpha = _alpha(percents, cols, y_offset + dy, t_now)

    return NoteOffsets(dx=dx, dy=dy, rotation_deg=rotation,
                       alpha_mult=alpha, zoom=zoom)


def receptor_offsets(percents: dict, cols: np.ndarray, t_now: float,
                     beat_now: float, keycount: int,
                     arrow_size: float = ARROW_SIZE,
                     rand_seed: int = 0) -> NoteOffsets:
    """`note_offsets` at y_offset = 0: receptors don't scroll, but drunk /
    tornado / tipsy still displace them, confusion still spins them, and
    mini/movex/movey still apply. dizzy uses note_beat = beat_now (no
    beats-until-step), so it vanishes, matching the engine (receptors use
    GetRotationZ at the current beat)."""
    cols = np.asarray(cols)
    y0 = np.zeros(cols.shape[0], dtype=np.float64)
    return note_offsets(percents, cols, y0, t_now, beat_now, keycount,
                        note_beats=np.full(cols.shape[0], beat_now),
                        arrow_size=arrow_size, rand_seed=rand_seed)
