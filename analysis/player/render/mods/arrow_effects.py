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

- confusion / confusionoffset: a whole-field Z spin keyed to the song beat
  ((beat*percent mod 2*PI)*-180/PI) plus a constant offset*180/PI; dizzy is
  the note-relative spin. Both implemented (`confusion_rotation` /
  `dizzy_rotation`). RotationX/Y confusion (confusionx / confusiony, with
  their *offset companions and numbered per-column variants) share that exact
  angle (`_confusion_axis_degrees`) but tilt about the horizontal / vertical
  axis. The SCALAR variants are a whole-field tilt the NotITG field
  projection renders as true perspective (games/notitg/field_projection
  sums them into the field model matrix); while it owns the tilt the
  consumer zeroes them here (note_mods' double-apply guard). The 2D
  foreshortening kernels below remain as the fallback where the
  projection cannot render - per-column numbered variants, the
  base-hidden deferral, instance-owned fields: confusionx -> uniform
  zoom by cos(angle) (`confusionx_zoom`, provably the fov->0 limit of
  the real tilt - TRANSFORM3D.md), confusiony -> a per-column dx pulling
  x toward center by (cos(angle)-1) (`confusiony_dx`).

- xmode: on a single-side field, dx = percent * y_offset - the vertical
  scroll shears into a diagonal (`xmode_x`); the doubles sign-split is not
  modeled.

- The DIGITAL / WAVEFORM WARP family (digital, zigzag, sawtooth, square,
  bounce) is a NotITG-era ArrowEffects extension absent from OpenITG's
  ArrowEffects.cpp; formulas are ported from ITGmania's ArrowEffects.cpp
  (the open fork that merged the NotITG mods) plus RageMath's RageTriangle
  / RageSquare. Each is a periodic sideways (X) shove of the same shape as
  drunk/beat: a wave of the note's y_offset, phase-shifted by a companion
  `<name>offset` percent and stretched by a `<name>period` percent, scaled
  to +/- ARROW_SIZE/2 (sawtooth to +/- ARROW_SIZE). `digital` additionally
  quantizes its sine to `digitalsteps + 1` levels. See `digital_x`,
  `zigzag_x`, `sawtooth_x`, `square_x`, `bounce_x` for the per-formula
  provenance. Their Z-axis siblings (digitalz, zigzagz, ...) push the note
  along +z in the engine; like bumpy we reproject that push to a per-note
  zoom (`waveform_z_zoom`). Companions ride the `percents` dict by name,
  the same way `_expand_phase` / `blink` do; per-column numbered variants
  ride the existing auto-detection.

- TAN / COSEC FAMILY: NotITG/ITGmania ship tan* siblings of the periodic
  mods (tandrunk, tantipsy, tantornado, tanbumpy, tanbumpyx, tandigital,
  and their *z forms) that swap the cos/sin kernel for tan (or 1/sin under
  the Cosecant flag) - the same math with a sharper, spikier profile. Ported
  via `_select_tan`; each shares its base formula's companions under a `tan`
  prefix. `tanexpand` is a scroll-speed multiplier that belongs in the host
  y_offset pipeline (like `expand`), so it is not a per-note mod here.

- BEAT SIBLINGS: beaty (Y/scroll axis) and beatz (Z -> zoom) accompany beat,
  all three sharing `beat_factor` (the once-per-beat pulse) with per-axis
  offset/period/mult companions (beatoffset/beatmult, beatyoffset/beatymult,
  beatzoffset/beatzmult). beatz reprojects to zoom like bumpy/digitalz.

- ATTENUATE / PARABOLA: attenuatex/y/z = percent*(yoff/AS)^2*(xoff/AS), a
  quadratic push scaled by the column's signed x-offset; parabolax/y/z drop
  the column term. X -> dx, Y -> dy, Z -> zoom (`attenuate`, `parabola`).

- PULSE / SHRINK (zoom): pulse (pulseinner/pulseouter + pulseoffset/period)
  is a per-note zoom pulse of the y_offset (`pulse_zoom`); shrinkmult /
  shrinklinear shrink approaching arrows by distance (`shrink_zoom`). Both
  land on the per-note zoom output now that the zoom hook exists.

- BOOMERANG: `accel_y_offset` applies the position parabola; the visibility
  half (engine peak/past-peak + culling, which we do not model) is expressed
  as an alpha fade past the fold (`boomerang_visibility`). See those two.

  DEFERRED: `grain` / `granulate` (a hold-body step-size multiplier, not a
  per-note position offset - it changes how many segments a hold is rendered
  with, a hold-render concern outside this module) and `dizzyholds` (a
  hold-render-specific spin). Both documented, not silently dropped.

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

from analysis.player.render.transform3d import eye_distance

ARROW_SIZE = 64.0
SCREEN_WIDTH = 640.0
SCREEN_HEIGHT = 480.0
CENTER_LINE_Y = 160.0
FADE_DIST_Y = 40.0
PI = np.pi

# LoadMenuPerspective camera distance (fov 45 over the design width):
# a +z push reads as the center-plane perspective scale d / (d - z).
EYE_DISTANCE = eye_distance(45.0, SCREEN_WIDTH)
# +z clamp keeping the perspective divide off the eye plane; a note
# pushed to the camera saturates at this scale instead of exploding.
_MAX_Z_SCALE = 32.0

# ITGmania metrics.ini [ArrowEffects] defaults (the values the formulas were
# tuned against; the engine reads them per-theme, we bake the _fallback set).
DRUNK_COLUMN_FREQUENCY = 0.2
DRUNK_OFFSET_FREQUENCY = 10.0
DRUNK_ARROW_MAGNITUDE = 0.5
TIPSY_TIMER_FREQUENCY = 1.2
TIPSY_COLUMN_FREQUENCY = 1.8
TIPSY_ARROW_MAGNITUDE = 0.4
TORNADO_OFFSET_FREQUENCY = 6.0
BUMPY_HEIGHT = 16.0
BEAT_OFFSET_HEIGHT = 15.0
BEAT_PI_HEIGHT = 2.0
WAVE_MOD_HEIGHT = 38.0
BOOMERANG_PEAK_PERCENTAGE = 0.75
EXPAND_MULTIPLIER_FREQUENCY = 3.0


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
    """Quantize (RageUtil.h:175): int((f + i/2)/i)*i. The int() cast
    TRUNCATES toward zero (C semantics), so np.trunc - not np.floor -
    matches it for the negative arguments blink's sine reaches."""
    return np.trunc((f + interval / 2.0) / interval).astype(np.float64) * interval


def _select_tan(angle, cosecant=False):
    """SelectTanType (ArrowEffects.cpp:209-215): the tan* family's periodic
    kernel - tan(angle) normally, 1/sin(angle) (cosecant) when the chart set
    the Cosecant flag. Replaces cos/sin in the drunk / tipsy / tornado / bumpy
    / digital formulas, giving the sharper `tan*` companions their spikes."""
    if cosecant:
        return 1.0 / np.sin(angle)
    return np.tan(angle)


def drunk_angle(cols, y_offset, t_now, speed, offset, period,
                col_freq=DRUNK_COLUMN_FREQUENCY,
                offset_freq=DRUNK_OFFSET_FREQUENCY):
    """CalculateDrunkAngle (ArrowEffects.cpp:242-249). The phase shared by
    drunk and its Z sibling. `speed` scales time (1+speed), `offset` scales
    the per-column term, `period` scales the y_offset term; all default to 0,
    recovering the plain `t + col*0.2 + yoff*10/H` phase. WALLCLOCK -> t_now."""
    col_idx = cols.astype(np.float64)
    return (t_now * (1.0 + speed)
            + col_idx * ((offset * col_freq) + col_freq)
            + y_offset * ((period * offset_freq) + offset_freq) / SCREEN_HEIGHT)


def drunk_x(percent, cols, y_offset, t_now, keycount, arrow_size=ARROW_SIZE,
            speed=0.0, offset=0.0, period=0.0, is_tan=False):
    """GetXPos drunk (ArrowEffects.cpp:854-864). WALLCLOCK -> t_now.
    speed/offset/period are the drunkspeed/drunkoffset/drunkperiod companions
    (default 0 = identity vs the previous fixed-frequency form). is_tan
    switches to the tandrunk kernel (tan/cosec, ArrowEffects.cpp:866-878)."""
    angle = drunk_angle(cols, y_offset, t_now, speed, offset, period)
    kernel = _select_tan(angle) if is_tan else np.cos(angle)
    return percent * (kernel * arrow_size * DRUNK_ARROW_MAGNITUDE)


def _tornado_window(cols, keycount, arrow_size, dimension):
    """The per-note [min_x, max_x] arccos window shared by tornado and
    tan-tornado (CalculateTornadoOffsetFromMagnitude, ArrowEffects.cpp:217).
    Window half-width narrows from 3 to 2 in wide fields (>4 cols) ONLY for
    dimension 0 (X); the Z window (dimension 2) keeps width 3 there
    (ArrowEffects::Init :358 `if (dimension == 0 && wide) width = 2`)."""
    xoffsets = column_offsets(keycount, arrow_size)
    width = 2 if (dimension == 0 and keycount > 4) else 3
    col_i = cols.astype(np.int64)
    start = np.clip(col_i - width, 0, keycount - 1)
    end = np.clip(col_i + width, 0, keycount - 1)

    min_x = np.empty(cols.shape, dtype=np.float64)
    max_x = np.empty(cols.shape, dtype=np.float64)
    for i in range(cols.shape[0]):
        window = xoffsets[start[i]:end[i] + 1]
        min_x[i] = window.min()
        max_x[i] = window.max()
    return xoffsets[col_i], min_x, max_x


def _tornado_offset(percent, cols, y_offset, keycount, arrow_size,
                    offset, period, is_tan, dimension=0):
    """CalculateTornadoOffsetFromMagnitude (ArrowEffects.cpp:217-239). The
    column's real x maps to [-1, 1] within its arccos window; (y_offset +
    offset) advances the phase at frequency (period+1)*6; cos (or tan/cosec
    for the tan variant) maps back to a windowed x. `offset`/`period` are the
    tornadooffset/tornadoperiod companions (default 0 = identity). `dimension`
    (0 = X, 2 = Z) selects the window width, wider on Z in a wide field."""
    real, min_x, max_x = _tornado_window(cols, keycount, arrow_size, dimension)
    span = np.where(max_x == min_x, 1.0, max_x - min_x)
    between = np.clip(_scale(real, min_x, min_x + span, -1.0, 1.0), -1.0, 1.0)
    freq = TORNADO_OFFSET_FREQUENCY
    rads = np.arccos(between) + (y_offset + offset) * ((period * freq) + freq) / SCREEN_HEIGHT
    processed = _select_tan(rads) if is_tan else np.cos(rads)
    adjusted = _scale(processed, -1.0, 1.0, min_x, max_x)
    return (adjusted - real) * percent


def tornado_x(percent, cols, y_offset, keycount, arrow_size=ARROW_SIZE,
              offset=0.0, period=0.0):
    """GetXPos tornado (ArrowEffects.cpp:820-826)."""
    return _tornado_offset(percent, cols, y_offset, keycount, arrow_size,
                           offset, period, is_tan=False)


def tan_tornado_x(percent, cols, y_offset, keycount, arrow_size=ARROW_SIZE,
                  offset=0.0, period=0.0):
    """GetXPos tantornado (ArrowEffects.cpp:828-834): tornado with the tan
    kernel (a sharper, spikier sway). Companions tantornadooffset /
    tantornadoperiod."""
    return _tornado_offset(percent, cols, y_offset, keycount, arrow_size,
                           offset, period, is_tan=True)


def reverse_fractions(percents: dict, cols: np.ndarray, keycount: int) -> np.ndarray:
    """Per-column reverse fraction r_col, ported from OpenITG
    PlayerOptions::GetReversePercentForColumn (PlayerOptions.cpp:539-562).

    The scroll family is a global reverse plus per-column numbered reverse
    plus three membership-gated percents, whose column membership matches
    the engine's integer tests exactly:
      - reverse   : every column.
      - reverse<c>: the single numbered column c (m_fReverse[iCol],
        PlayerOptions.cpp:1688), added alongside the global reverse.
      - split     : columns in the RIGHT half (iCol >= N/2, integer div).
      - alternate : ODD columns (iCol % 2 == 1).
      - cross     : the MIDDLE half [N/4, N-1-N/4] inclusive.
    The raw sum is wrapped the engine's way: > 2 folds mod 2, then a value
    in (1, 2] mirrors back down via SCALE(f, 1,2, 1,0) so r stays in [0, 1]
    (a note fully reversed twice reads as un-reversed). Returns one r per
    note aligned with `cols`; `centered` is handled separately (it is a
    SCROLL_CENTERED shift, not part of the per-column reverse percent).

    Odd keycounts: N/2 and N/4 are integer divisions exactly as in C, so
    e.g. N=5 -> split covers cols {2,3,4}, cross covers [1, 3]; the middle
    column of an odd field is whatever those integer bounds include."""
    reverse = float(percents.get('reverse', 0.0))
    split = float(percents.get('split', 0.0))
    alternate = float(percents.get('alternate', 0.0))
    cross = float(percents.get('cross', 0.0))

    reverse_col = np.full(keycount, reverse, dtype=np.float64)
    for c in range(keycount):
        reverse_col[c] += float(percents.get(f'reverse{c}', 0.0))

    col_i = cols.astype(np.int64)
    f = reverse_col[col_i]
    f += np.where(col_i >= keycount // 2, split, 0.0)
    f += np.where((col_i % 2) == 1, alternate, 0.0)

    first_cross = keycount // 4
    last_cross = keycount - 1 - first_cross
    in_cross = (col_i >= first_cross) & (col_i <= last_cross)
    f += np.where(in_cross, cross, 0.0)

    f = np.where(f > 2.0, np.mod(f, 2.0), f)
    f = np.where(f > 1.0, _scale(f, 1.0, 2.0, 1.0, 0.0), f)
    return f


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


def beat_factor(beat_now, offset=0.0, mult=0.0):
    """UpdateBeat (ArrowEffects.cpp:260-298): the per-frame beat pulse factor
    (already scaled by 20). A triangle-ish pulse fires once per beat window;
    `offset` shifts the beat phase (beatoffset), `mult` speeds the pulse
    ((mult+1)x, beatmult). Same factor drives beat / beaty / beatz - only the
    sin() shift below differs by axis. Returns a scalar."""
    accel_time, total_time = 0.2, 0.5
    beat = (beat_now + accel_time + offset) * (mult + 1.0)
    even = (int(beat) % 2) != 0

    if beat < 0.0:
        return 0.0
    beat -= np.trunc(beat)
    beat += 1.0
    beat -= np.trunc(beat)
    if beat >= total_time:
        return 0.0

    if beat < accel_time:
        amount = _scale(beat, 0.0, accel_time, 0.0, 1.0)
        amount *= amount
    else:
        amount = _scale(beat, accel_time, total_time, 1.0, 0.0)
        amount = 1.0 - (1.0 - amount) * (1.0 - amount)
    if even:
        amount *= -1.0
    return 20.0 * amount


def _beat_shift(factor, y_offset, period):
    """The sin() term shared by beat / beaty / beatz (ArrowEffects.cpp:898-905
    etc.): factor * sin(yoff/((period*15)+15) + PI/2)."""
    height = (period * BEAT_OFFSET_HEIGHT) + BEAT_OFFSET_HEIGHT
    return factor * np.sin(y_offset / height + PI / BEAT_PI_HEIGHT)


def beat_x(percent, y_offset, beat_now, offset=0.0, period=0.0, mult=0.0):
    """GetXPos beat (ArrowEffects.cpp:897-906): a periodic sideways shove keyed
    to the song beat. Companions beatoffset / beatperiod / beatmult."""
    return percent * _beat_shift(beat_factor(beat_now, offset, mult),
                                 y_offset, period)


def beat_y(percent, y_offset, beat_now, offset=0.0, period=0.0, mult=0.0):
    """GetYPos beaty (ArrowEffects.cpp:762-771): the beat pulse on the Y
    (scroll) axis. Companions beatyoffset / beatyperiod / beatymult."""
    return percent * _beat_shift(beat_factor(beat_now, offset, mult),
                                 y_offset, period)


def beat_z(percent, y_offset, beat_now, offset=0.0, period=0.0, mult=0.0):
    """GetZPos beatz (ArrowEffects.cpp:1481-1489): the beat pulse on Z. Like
    bumpy/digitalz it is a +z push in engine px; reproject to zoom via
    `waveform_z_zoom`. Companions beatzoffset / beatzperiod / beatzmult."""
    return percent * _beat_shift(beat_factor(beat_now, offset, mult),
                                 y_offset, period)


def xmode_x(percent, y_offset):
    """GetXPos xmode (ArrowEffects.cpp:990-1019). For a single-side field
    (our only style) it is simply percent * yOffset: the further a note is
    from the receptor, the more it is shoved sideways, turning the vertical
    scroll into a diagonal (an "X" with reverse). The multi-side sign split
    (P2 / right-half columns get -yOffset) is a doubles-only concern we do not
    model. Returns a per-note dx."""
    return percent * np.asarray(y_offset, dtype=np.float64)


def movex_x(percent, arrow_size=ARROW_SIZE):
    """NotITG movex: 100% = one arrow width along x. `percent` is a
    per-note array (per-column variants handled upstream)."""
    return percent * arrow_size


def rage_triangle(angle):
    """Vectorized port of RageMath.cpp RageTriangle: a triangle wave in
    [-1, 1] with period 2*PI. The angle is wrapped to [0, 2*PI); u =
    angle/PI in [0, 2); the wave rises 0->1 over u in [0, 0.5), falls
    1->-1 over [0.5, 1.5), rises -1->0 over [1.5, 2)."""
    a = np.mod(np.asarray(angle, dtype=np.float64), 2.0 * PI)
    u = a / PI
    rising_front = u * 2.0
    falling = 1.0 - (u - 0.5) * 2.0
    rising_back = -4.0 + u * 2.0
    return np.where(u < 0.5, rising_front,
                    np.where(u < 1.5, falling, rising_back))


def rage_square(angle):
    """Vectorized port of RageMath.cpp RageSquare (RageMath.cpp:584): a
    square wave, -1 for the second half of each 2*PI period, +1 for the
    first. The engine's <0.01 guard nudges a near-zero wrapped angle up by
    2*PI (a hold-flicker hack); for a small POSITIVE angle in [0, 0.01)
    that pushes it past PI, flipping the result from +1 to -1, so it is NOT
    a no-op - a note whose phase lands in that band gets the opposite
    sign."""
    a = np.mod(np.asarray(angle, dtype=np.float64), 2.0 * PI)
    a = np.where(a < 0.01, a + 2.0 * PI, a)
    return np.where(a >= PI, -1.0, 1.0)


def _digital_angle(y_offset, offset, period, arrow_size):
    """CalculateDigitalAngle (ITGmania ArrowEffects.cpp:255): the sine
    phase shared by digital and square. `offset` shifts by 1 engine px per
    percent; `period` stretches the ARROW_SIZE-wide base period."""
    return PI * (y_offset + offset) / (arrow_size + period * arrow_size)


def digital_x(percent, y_offset, offset, period, steps, arrow_size=ARROW_SIZE,
              is_tan=False):
    """digital (ITGmania ArrowEffects.cpp:943-952): a sine shove of the
    note's y_offset quantized to `steps + 1` discrete levels (the
    square-stepped sine the docs describe). `percent` is per note (numbered
    per-column variants upstream); `offset`/`period`/`steps` are scalar
    companion percents. is_tan switches to the tandigital kernel (tan/cosec),
    ArrowEffects.cpp:954-966.

    levels = steps + 1; contribution = percent * ARROW_SIZE * 0.5 *
    round(levels * sin(angle)) / levels. At steps = 0 the round/levels
    collapse to round(sin) in {-1, 0, 1} (a coarse three-level staircase);
    larger steps recover a smoother sine."""
    levels = steps + 1.0
    angle = _digital_angle(y_offset, offset, period, arrow_size)
    kernel = _select_tan(angle) if is_tan else np.sin(angle)
    quantized = np.round(levels * kernel) / levels
    return percent * arrow_size * 0.5 * quantized


def zigzag_x(percent, y_offset, offset, period, arrow_size=ARROW_SIZE):
    """zigzag (ITGmania ArrowEffects.cpp:908-917): a triangle wave of the
    note's y_offset, scaled to +/- ARROW_SIZE/2. `offset` shifts the phase
    by 100 engine px per percent; `period` stretches the wave (1/(period+1)
    frequency multiplier)."""
    angle = PI * (1.0 / (period + 1.0)) * ((y_offset + 100.0 * offset) / arrow_size)
    return percent * (arrow_size / 2.0) * rage_triangle(angle)


def sawtooth_x(percent, y_offset, period, arrow_size=ARROW_SIZE):
    """sawtooth (ITGmania ArrowEffects.cpp:919-928): a rising sawtooth, the
    fractional part of a y_offset ramp, scaled to ARROW_SIZE. `period`
    stretches the ramp (0.5/(period+1) slope). No offset companion in the
    engine formula (the docs list SawtoothOffset but the ported source does
    not read it), so it is omitted; documented in `note_offsets`."""
    ramp = (0.5 / (period + 1.0) * y_offset) / arrow_size
    return percent * arrow_size * (ramp - np.floor(ramp))


def square_x(percent, y_offset, offset, period, arrow_size=ARROW_SIZE):
    """square (ITGmania ArrowEffects.cpp:970-981): a square wave of the
    note's y_offset, scaled to +/- ARROW_SIZE/2. Shares the digital phase
    (offset = 1 engine px per percent, period stretches ARROW_SIZE)."""
    angle = _digital_angle(y_offset, offset, period, arrow_size)
    return percent * arrow_size * 0.5 * rage_square(angle)


def bounce_x(percent, y_offset, offset, period, arrow_size=ARROW_SIZE):
    """bounce (ITGmania ArrowEffects.cpp:983-993): a rectified sine
    (abs(sin)) of the note's y_offset - arrows bounce toward the receptors
    always on one side, scaled to ARROW_SIZE/2. Base period 60 engine px,
    stretched by `period`; `offset` shifts by 1 engine px per percent."""
    amt = np.abs(np.sin((y_offset + offset) / (60.0 + period * 60.0)))
    return percent * arrow_size * 0.5 * amt


def parabola(percent, y_offset, arrow_size=ARROW_SIZE):
    """parabolax/y/z (ArrowEffects.cpp:931-934, 638-641, 1445-1448):
    percent * (yoff/AS)^2. A quadratic push whose axis is chosen by the caller
    (X = dx, Y = dy, Z = z-push). No column term (unlike attenuate)."""
    r = np.asarray(y_offset, dtype=np.float64) / arrow_size
    return percent * r * r


def attenuate(percent, cols, y_offset, keycount, arrow_size=ARROW_SIZE):
    """attenuatex/y/z (ArrowEffects.cpp:936-940, 756-759, 1450-1454):
    percent * (yoff/AS)^2 * (xoff/AS). Like parabola but scaled by the column's
    signed x-offset, so the push grows with distance from field center and
    flips sign across the center column. Axis chosen by the caller."""
    xoff = column_offsets(keycount, arrow_size)[cols.astype(np.int64)]
    r = np.asarray(y_offset, dtype=np.float64) / arrow_size
    return percent * r * r * (xoff / arrow_size)


def perspective_z_scale(z_push):
    """The center-plane perspective scale of an engine +z push:
    d / (d - z) with d the LoadMenuPerspective eye distance (EYE_DISTANCE,
    transform3d.eye_distance). This is the exact scale the real projection
    gives a z-translated plane at the design center - the sanctioned
    per-note degradation of 3D (TRANSFORM3D.md, "What each executor
    consumes"). A push at/behind the eye saturates at _MAX_Z_SCALE
    instead of crossing the divide."""
    z = np.minimum(z_push, EYE_DISTANCE * (1.0 - 1.0 / _MAX_Z_SCALE))
    return EYE_DISTANCE / (EYE_DISTANCE - z)


def waveform_z_zoom(z_push):
    """Reproject an engine +z push (the waveform Z siblings digitalz /
    zigzagz / sawtoothz / squarez / bouncez accumulate into fZPos) to a
    2D zoom multiplier: the center-plane perspective scale, exactly as
    `bumpy_zoom` does. `z_push` is the summed per-note z contribution in
    engine px."""
    return perspective_z_scale(z_push)


def accel_y_offset(percents: dict, y_offset: np.ndarray,
                   field_height: float = SCREEN_HEIGHT) -> np.ndarray:
    """Reshape the pre-mod y_offset per ArrowEffects::GetYOffset's accel
    section (ArrowEffects.cpp:64-84, 123-135). Returns a NEW y_offset the
    position pipeline should use in place of the raw one.

    Ported exactly:
      - BOOST (:64-72): notes bunch far away and accelerate in. fNewY =
        y * 1.5 / ((y + H/1.2) / H); adjust = boost * (fNewY - y), clamped
        [-400, 400].
      - BRAKE (:73-82): notes slow near the receptor. scale = y/H;
        fNewY = y * scale; adjust = brake * (fNewY - y), clamped [-400, 400].
      - WAVE (:83-84): adjust += wave * 20 * sin(y / (waveperiod*38 + 38)),
        the waveperiod companion stretching the spatial frequency
        (WAVE_MOD_HEIGHT = 38; default period 0 -> /38).
      - EXPAND (:123-133): a periodic SCROLL-SPEED multiplier (cos of a
        wall-clock timer). WALLCLOCK -> we key it to song time via the
        `_expand_phase` percent the caller injects (radians); multiplier =
        SCALE(cos(phase), -1,1, 0.75,1.75), applied as
        SCALE(expand, 0,1, 1, mult) scaling the whole offset.

    Notes past the receptor (y < 0) are returned untouched for the WHOLE
    accel section: the engine's GetYOffset early-out (:595 `if (fYOffset <
    0) return fYOffset * fScrollSpeed;`) returns before boost/brake/wave,
    before the boomerang fold, and before the expand speed multiply, so a
    past note keeps its raw offset. We match that with a final
    `np.where(past, y, out)` rather than folding y<0 notes.

    BOOMERANG (:646-655) now IMPLEMENTED. It rewrites the (already
    accel-adjusted) offset into a downward parabola
        y' = -y*y/H + 1.5*y
    so an arrow rises from the receptor, decelerates to a peak, then falls
    back - the "boomerang" throw. The engine also emits fPeakYOffsetOut =
    -p*p/H + 1.5*p at p = H*0.75 and bIsPastPeakOut = y < p, feeding its
    culling / gray-arrow logic. Our culling reads RAW scroll distance and has
    no peak contract, so we express the visibility half of boomerang here as
    an alpha companion (see `boomerang_visibility`): an arrow whose raw offset
    is beyond the fold has "boomeranged back" and the engine would have culled
    or grayed it, so we fade it. The position parabola runs for every
    APPROACHING note (y>=0); the y<0 early-out above precedes it, so past
    notes are never folded."""
    boost = float(percents.get('boost', 0.0))
    brake = float(percents.get('brake', 0.0))
    wave = float(percents.get('wave', 0.0))
    expand = float(percents.get('expand', 0.0))
    boomerang = float(percents.get('boomerang', 0.0))
    if not (boost or brake or wave or expand or boomerang):
        return y_offset

    y = np.asarray(y_offset, dtype=np.float64)
    past = y < 0.0
    adjust = np.zeros_like(y)

    if boost:
        new_y = y * 1.5 / ((y + field_height / 1.2) / field_height)
        adjust += np.clip(boost * (new_y - y), -400.0, 400.0)
    if brake:
        scale = _scale(y, 0.0, field_height, 0.0, 1.0)
        new_y = y * scale
        adjust += np.clip(brake * (new_y - y), -400.0, 400.0)
    if wave:
        wave_period = float(percents.get('waveperiod', 0.0))
        adjust += wave * 20.0 * np.sin(
            y / ((wave_period * WAVE_MOD_HEIGHT) + WAVE_MOD_HEIGHT))

    out = y + adjust
    if boomerang:
        out = boomerang_y_offset(out, field_height)
    if expand:
        phase = float(percents.get('_expand_phase', 0.0))
        mult = _scale(np.cos(phase), -1.0, 1.0, 0.75, 1.75)
        out = out * _scale(expand, 0.0, 1.0, 1.0, mult)

    # Engine early-out (:595): a past-receptor note (y<0) returns its raw
    # offset before boost/brake/wave, the boomerang fold, AND the expand
    # multiply. Restore it here so every past note keeps y untouched.
    return np.where(past, y, out)


def boomerang_y_offset(y_offset, field_height: float = SCREEN_HEIGHT):
    """The boomerang position parabola (ArrowEffects.cpp:654):
    y' = -y*y/H + 1.5*y. A note rises, peaks at y = 0.75*H (fold point), and
    falls back. Pure per-note reshape of the offset."""
    y = np.asarray(y_offset, dtype=np.float64)
    return -1.0 * y * y / field_height + 1.5 * y


def boomerang_peak(field_height: float = SCREEN_HEIGHT):
    """fPeakYOffsetOut (ArrowEffects.cpp:650-651): the post-parabola offset of
    the fold, at the raw peak p = H * 0.75. Returns (peak_raw, peak_y') so a
    caller can reason about where notes turn around."""
    p = field_height * BOOMERANG_PEAK_PERCENTAGE
    peak_y = -1.0 * p * p / field_height + 1.5 * p
    return p, peak_y


def boomerang_visibility(percent, y_offset, field_height: float = SCREEN_HEIGHT):
    """Visibility approximation for boomerang (the half the engine handles via
    bIsPastPeakOut + culling, which we do not model).

    Study of the engine: with boomerang on, an arrow's raw offset climbs past
    the fold p = H*0.75; bIsPastPeakOut = (rawY < p) marks arrows still on the
    way UP (not yet folded). Arrows with rawY > p have crossed the peak and are
    "boomeranging back" toward the receptor from the far side - the engine's
    draw-distance culling (fDrawDistanceBeforeTargetsPixels) and gray-arrow
    fade hide them past that fold rather than drawing a second pass on screen.

    We cannot replicate culling here, so we pick the closest EXPRESSIBLE
    behavior: fade an arrow to alpha 0 once its raw offset is beyond the fold,
    ramped over one ARROW_SIZE so the turn isn't a hard pop. `percent` gates the
    strength (boomerang 0 => no fade). Returns a per-note alpha multiplier in
    [0, 1]. This mirrors bIsPastPeakOut: visible while rawY <= p, fading after."""
    if percent == 0.0:
        return 1.0
    y = np.asarray(y_offset, dtype=np.float64)
    p = field_height * BOOMERANG_PEAK_PERCENTAGE
    fade = np.clip(_scale(y, p, p + ARROW_SIZE, 1.0, 0.0), 0.0, 1.0)
    return 1.0 - percent * (1.0 - fade)


def _tipsy_angle(cols, t_now, speed, offset):
    """The tipsy phase (UpdateTipsy, ArrowEffects.cpp:300-333). speed scales
    the timer term (1.2*(speed+1)), offset scales the per-column term
    (1.8*(offset+1)); both default 0. WALLCLOCK -> t_now."""
    col_idx = cols.astype(np.float64)
    return (t_now * ((speed * TIPSY_TIMER_FREQUENCY) + TIPSY_TIMER_FREQUENCY)
            + col_idx * ((offset * TIPSY_COLUMN_FREQUENCY) + TIPSY_COLUMN_FREQUENCY))


def tipsy_y(percent, cols, t_now, arrow_size=ARROW_SIZE, speed=0.0, offset=0.0,
            is_tan=False):
    """GetYPos tipsy (ArrowEffects.cpp:752-754 + UpdateTipsy). Columns bob
    along the scroll axis. speed/offset are the tipsyspeed/tipsyoffset
    companions; is_tan switches to the tantipsy kernel (tan/cosec)."""
    angle = _tipsy_angle(cols, t_now, speed, offset)
    kernel = _select_tan(angle) if is_tan else np.cos(angle)
    return percent * (kernel * arrow_size * TIPSY_ARROW_MAGNITUDE)


def movey_y(percent, arrow_size=ARROW_SIZE):
    """NotITG movey: 100% = one arrow width along y."""
    return percent * arrow_size


def _bumpy_angle(y_offset, offset, period):
    """CalculateBumpyAngle (ArrowEffects.cpp:251-253): (yoff + 100*offset) /
    ((period*16)+16). offset/period are the bumpyoffset/bumpyperiod (or
    bumpyxoffset/bumpyxperiod) companions; both default 0 -> yoff/16."""
    return (y_offset + 100.0 * offset) / ((period * BUMPY_HEIGHT) + BUMPY_HEIGHT)


def bumpy_z(percent, y_offset, offset=0.0, period=0.0, is_tan=False):
    """GetZPos bumpy (ArrowEffects.cpp:1399-1403): the raw engine +z push,
    percent*40*sin(angle). is_tan switches to the tanbumpy kernel. Returns
    engine px (reproject to zoom via `waveform_z_zoom` / `bumpy_zoom`)."""
    angle = _bumpy_angle(y_offset, offset, period)
    kernel = _select_tan(angle) if is_tan else np.sin(angle)
    return percent * 40.0 * kernel


def bumpy_zoom(percent, y_offset, offset=0.0, period=0.0, is_tan=False):
    """GetZPos bumpy reprojected to 2D zoom.

    In 3D, fZPos += percent * 40 * sin(bumpy_angle); +z is toward the
    camera => larger, as the center-plane perspective scale
    (`perspective_z_scale`). offset/period/is_tan carry the bumpyoffset/
    bumpyperiod/tanbumpy companions."""
    return perspective_z_scale(bumpy_z(percent, y_offset, offset, period, is_tan))


def bumpy_x(percent, y_offset, offset=0.0, period=0.0, is_tan=False):
    """GetXPos bumpyx (ArrowEffects.cpp:836-842): bumpy applied on the X axis
    - the same 40*sin(bumpy_angle) shove sideways instead of into z. This one
    is a true per-note dx (no reprojection). Companions bumpyxoffset /
    bumpyxperiod; is_tan = tanbumpyx."""
    angle = _bumpy_angle(y_offset, offset, period)
    kernel = _select_tan(angle) if is_tan else np.sin(angle)
    return percent * 40.0 * kernel


def dizzy_rotation(percent, note_beat, beat_now):
    """GetRotationZ dizzy (ArrowEffects.cpp:364-378): spin proportional to
    beats-until-step, wrapped to a full turn, in degrees. `note_beat` is
    the note's own beat, `beat_now` the current song beat."""
    rot = (note_beat - beat_now) * percent
    rot = np.mod(rot, 2.0 * PI)
    return rot * 180.0 / PI


def _confusion_axis_degrees(percent, beat_now, offset):
    """The shared confusion angle in degrees, identical on all three axes:
    ReceptorGetRotationZ/X/Y (ArrowEffects.cpp:1091-1113 / 1115-1138 /
    1140-1163) each compute the SAME (songBeat*percent wrapped to 2*PI)*-180/PI
    spin plus a constant offset*180/PI. Only the axis the result rotates about
    differs. `percent` is the confusion/confusionx/confusiony magnitude,
    `offset` its matching confusion*offset companion."""
    spin = np.mod(beat_now * percent, 2.0 * PI) * -180.0 / PI
    return spin + offset * 180.0 / PI


def confusion_rotation(percent, beat_now, offset=0.0):
    """confusion / confusionoffset (ReceptorGetRotationZ,
    ArrowEffects.cpp:1091-1110): a whole-field Z spin (in-plane). Returned in
    degrees, one scalar broadcast to every note. See `_confusion_axis_degrees`
    for the formula shared with the X/Y siblings."""
    return _confusion_axis_degrees(percent, beat_now, offset)


def confusionx_zoom(percent, beat_now, offset=0.0):
    """confusionx / confusionxoffset reprojected to 2D zoom.

    ReceptorGetRotationX (ArrowEffects.cpp:1115-1138) rotates the whole field
    about the HORIZONTAL axis by the confusion angle (the X-axis sibling of the
    Z-axis confusion spin; same magnitude, different axis). A rotation about X
    tilts the field toward/away from the camera, which in true 3D foreshortens
    the VERTICAL extent by cos(angle).

    Our pipeline exposes only a UNIFORM per-note zoom (renderer scales x and y
    by the same factor; there is no independent zoom_y), so we express the X
    tilt as that uniform zoom multiplied by cos(angle): a note tilted flat
    about X reads smaller. This is the closest faithful uniform-zoom projection
    of an out-of-plane X tilt, in the same spirit as `bumpy_zoom`'s z->zoom
    proxy. abs() keeps the multiplier non-negative past a quarter turn (a
    tilt past 90 deg reads as shrinking back to edge-on, never mirrored).
    Returns a scalar zoom multiplier broadcast to every note."""
    angle = _confusion_axis_degrees(percent, beat_now, offset) * PI / 180.0
    return np.abs(np.cos(angle))


def confusiony_dx(percent, cols, beat_now, keycount, offset=0.0,
                  arrow_size=ARROW_SIZE):
    """confusiony / confusionyoffset reprojected to a 2D per-column dx.

    ReceptorGetRotationY (ArrowEffects.cpp:1140-1163) rotates the whole field
    about the VERTICAL axis by the confusion angle (the Y-axis sibling; same
    magnitude as confusion/confusionx). A rotation about Y foreshortens the
    HORIZONTAL extent by cos(angle): every column's x-offset from field center
    contracts toward the center (the vanishing line) as the field tilts.

    We render 2D, so we express that horizontal foreshortening as a per-column
    dx that pulls each column's x-offset toward center by (cos(angle) - 1):
        dx = xoff * (cos(angle) - 1)
    At angle 0 the field is head-on (cos 1, dx 0); at a quarter turn the field
    is seen edge-on (cos 0, columns collapse onto center). This is the
    horizontal-foreshortening analogue of `confusionx_zoom`'s vertical one,
    chosen because the pipeline has per-note dx but no independent zoom_x.
    Returns a per-note dx aligned with `cols`."""
    angle = _confusion_axis_degrees(percent, beat_now, offset) * PI / 180.0
    xoff = column_offsets(keycount, arrow_size)[cols.astype(np.int64)]
    return xoff * (np.cos(angle) - 1.0)


def hallway_x(percent, cols, y_offset, keycount, arrow_size=ARROW_SIZE,
              field_height=SCREEN_HEIGHT):
    """hallway: a per-note X scale toward the vanishing point with distance.

    NotITG's `hallway` is a perspective/appearance mod (a notefield tilt, like
    distant/space/incoming/overhead) rather than a GetXPos term, so it has no
    formula in OpenITG/ITGmania's ArrowEffects.cpp - the perspective is applied
    at the notefield-actor level (m_fPerspectiveTilt, OpenITG ArrowEffects.cpp
    :17). Its VISUAL effect is exactly 2D-expressible as a per-note dx: notes
    far from the receptor recede toward the field-center vanishing line, notes
    at the receptor stay put.

    We port that effect directly. An approaching note (y_offset > 0) at column
    x-offset `xoff` and depth `y_offset` has its x contracted toward center by
    a pinhole perspective factor f = H / (H + y_offset) in (0, 1]:
        dx = xoff * (f - 1) * percent
    At the receptor (y_offset 0) f = 1 => dx 0; far away f -> 0 => the column
    collapses onto center (the vanishing point). Notes past the receptor
    (y_offset < 0) are left unscaled (they are at/through the judgment line,
    not receding down the hallway). Returns a per-note dx aligned with `cols`."""
    xoff = column_offsets(keycount, arrow_size)[cols.astype(np.int64)]
    y = np.asarray(y_offset, dtype=np.float64)
    depth = np.maximum(y, 0.0)
    factor = field_height / (field_height + depth)
    return xoff * (factor - 1.0) * percent


def _center_line(mini_percent):
    # ArrowEffects.cpp GetCenterLine: CENTER_LINE_Y / (1 - mini*0.5).
    # The engine does the raw C++ float divide, so mini == 200% (zoom 0)
    # yields +inf, not a crash - the fade math downstream then treats the
    # visibility window as infinitely far (mini this extreme collapses
    # the field to a point). Match that instead of raising.
    zoom = 1.0 - mini_percent * 0.5
    if zoom == 0.0:
        return np.inf
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

    adjust = np.zeros(cols.shape, dtype=np.float64)
    # An extreme mini (>= 200%) drives the center line to +inf (the field
    # collapses to a point); the hidden/sudden fade windows go infinitely
    # far, so they never fade a note. Skip them rather than propagate
    # inf-inf NaN through the _scale ramps.
    if np.isfinite(center):
        hidden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -1.0, -1.25) + center * hidden_off
        hidden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 0.0, -0.25) + center * hidden_off
        sudden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -0.0, 0.25) + center * sudden_off
        sudden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 1.0, 1.25) + center * sudden_off
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
    compositing (the port-parity contract pins this smooth value);
    `display_alpha` maps it to the engine's on-screen brightness at the
    draw boundary."""
    return visible


def display_alpha(visible):
    """Perceived note brightness for one drawn part, from its raw
    visibility. The engine draws each note twice: GetAlpha is a HARD 0/1
    cut at visible > 0.5, and GetGlow (ArrowEffects.cpp:496-505) adds a
    glow pass ramping to 1.3 at the boundary - a fading note reads as
    bright glow well before its alpha flips (2.6x the raw visibility
    below the cut). One multiplier for our single-pass composite: full
    above the cut, the glow ramp below."""
    return np.clip(2.6 * np.asarray(visible, dtype=np.float64), 0.0, 1.0)


def zoom_from_mini(mini_percent):
    """GetZoom / mini (ArrowEffects.cpp:389): 100% mini = half size, 200%
    = zero. Returns a scalar zoom."""
    return 1.0 - mini_percent * 0.5


def tiny_zoom(tiny_percent):
    """NotITG tiny sprite zoom (GetZoom, ArrowEffects.cpp:1582): zoom *=
    pow(0.5, tiny), so 100% halves, 200% quarters, -100% doubles. This is
    a DIFFERENT curve from mini's 1 - p*0.5 (they agree only at 0 and 1);
    tiny additionally compresses the X spacing (see `tiny_spacing`), which
    mini does NOT."""
    return np.power(0.5, tiny_percent)


def tiny_spacing(tiny_percent):
    """NotITG tiny X-spacing compression (GetXPos, ArrowEffects.cpp:1025):
    the engine multiplies the WHOLE accumulated x offset - the summed mods
    AND the column's own x-offset from field center - by
    min(pow(0.5, tiny), 1). So POSITIVE tiny pulls the columns toward center
    (tighter spacing) as well as shrinking each sprite, while negative tiny
    (arrows grow) is GATED at 1 and leaves spacing unchanged. Returns the
    scalar column-space multiplier (1.0 when tiny is 0)."""
    if tiny_percent == 0.0:
        return 1.0
    return min(float(np.power(0.5, tiny_percent)), 1.0)


def pulse_zoom(inner, outer, y_offset, offset=0.0, period=0.0,
               arrow_size=ARROW_SIZE):
    """GetZoomVariable pulse (ArrowEffects.cpp:1596-1610 + GetPulseInner
    :1630-1641): a per-note zoom that pulses with y_offset. `outer` sets the
    swing amplitude, `inner` the rest scale; pulseoffset shifts the phase,
    pulseperiod stretches it. When both are 0 the mod is off (returns 1.0).

        sine = sin( (yoff + 100*offset) / (0.4*(AS + period*AS)) )
        pulse_inner = inner*0.5 + 1        (clamped away from exactly 0)
        zoom_mult = sine * (outer*0.5) + pulse_inner

    Returns a per-note zoom multiplier."""
    if inner == 0.0 and outer == 0.0:
        return 1.0
    height = 0.4 * (arrow_size + period * arrow_size)
    sine = np.sin((np.asarray(y_offset, dtype=np.float64) + 100.0 * offset) / height)
    pulse_inner = inner * 0.5 + 1.0
    if pulse_inner == 0.0:
        pulse_inner = 0.01
    return sine * (outer * 0.5) + pulse_inner


def shrink_zoom(shrink_mult, shrink_linear, y_offset, arrow_size=ARROW_SIZE):
    """GetZoomVariable shrink family (ArrowEffects.cpp:1611-1626), applied only
    to arrows still approaching (y_offset >= 0):
      - shrinkmult   (shrink_to_multiply): zoom *= 1/(1 + yoff*(mult/100)).
      - shrinklinear (shrink_to_linear):   zoom += yoff*(0.5*linear/AS).
    Returns a per-note (mult, add) pair so the caller can fold multiply before
    add, matching the engine order. Notes with y_offset < 0 are unaffected."""
    y = np.asarray(y_offset, dtype=np.float64)
    approaching = y >= 0.0
    mult = np.ones_like(y)
    add = np.zeros_like(y)
    if shrink_mult != 0.0:
        mult = np.where(approaching, 1.0 / (1.0 + y * (shrink_mult / 100.0)), 1.0)
    if shrink_linear != 0.0:
        add = np.where(approaching, y * (0.5 * shrink_linear / arrow_size), 0.0)
    return mult, add


def receptor_alpha_from_dark(dark_percent):
    """NotITG dark: hides the RECEPTORS (docs: "Hides the receptors, while
    keeping the ... flashes when tapping"). It is receptor-only, so it
    never enters note visibility; the receptor layer multiplies its mark
    alpha by this. 100% dark = invisible receptors, clamped to [0, 1]."""
    return float(np.clip(1.0 - dark_percent, 0.0, 1.0))


@dataclass(frozen=True)
class NoteOffsets:
    """Summed per-note contributions (arrays aligned with `cols`).

    dx/dy in pixels, rotation_deg the in-plane Z spin, alpha/zoom
    multipliers. The 3D fields carry the real per-note depth + out-of-
    plane tilt for the perspective note path: `z` the engine +z push
    (GetZPos - bumpy/digitalz/beatz...), `rot_x`/`rot_y` the roll/twirl
    tilts (degrees). They rest at 0, so a note with no depth/tilt keeps
    the flat 2D dx/dy/zoom/rotation draw. `zoom` holds only the
    genuinely-2D scale mods (mini/tiny/pulse/shrink) - the z push is no
    longer folded into it here (the projection does that)."""
    dx: np.ndarray
    dy: np.ndarray
    rotation_deg: np.ndarray
    alpha_mult: np.ndarray
    zoom: np.ndarray
    z: np.ndarray = None
    rot_x: np.ndarray = None
    rot_y: np.ndarray = None


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


def _drunk_pair(percents, cols, y_offset, t_now, keycount, arrow_size, suffix):
    """The drunk + tandrunk pair for one axis (suffix '' for X, 'z' for Z).
    Both share `drunk_x`; the tan sibling flips the kernel. Companions are
    speed/offset/period under the base name."""
    plain, tan = 'drunk' + suffix, 'tandrunk' + suffix
    out = np.zeros(cols.shape[0], dtype=np.float64)
    for name, is_tan in ((plain, False), (tan, True)):
        if _active(percents, name, keycount):
            out += drunk_x(_per_note(percents, name, cols, keycount), cols, y_offset,
                           t_now, keycount, arrow_size, _get(percents, name + 'speed'),
                           _get(percents, name + 'offset'), _get(percents, name + 'period'),
                           is_tan=is_tan)
    return out


def _tornado_pair(percents, cols, y_offset, keycount, arrow_size, suffix):
    """The tornado + tantornado pair for one axis. Companions offset/period.
    The engine window width depends on the axis (dimension 0 = X, 2 = Z), so
    the Z pair keeps width 3 in a wide field where X narrows to 2."""
    plain, tan = 'tornado' + suffix, 'tantornado' + suffix
    dimension = 2 if suffix == 'z' else 0
    out = np.zeros(cols.shape[0], dtype=np.float64)
    for name, is_tan in ((plain, False), (tan, True)):
        if _get(percents, name):
            out += _tornado_offset(_get(percents, name), cols, y_offset, keycount,
                                   arrow_size, _get(percents, name + 'offset'),
                                   _get(percents, name + 'period'), is_tan,
                                   dimension=dimension)
    return out


def _confusion_offset(percents, name, cols, keycount):
    """The per-note constant confusion offset for one axis: the global
    `<name>offset` companion plus any numbered per-column variants (`<name>0`
    ..), which the engine stores as m_fConfusion*[iCol] and adds as a per-column
    constant rotation (ReceptorGetRotationX/Y ArrowEffects.cpp:1120/1145). The
    global spin (`<name>`) rides separately as the beat term. Returns a per-note
    offset array aligned with `cols` (scalar-broadcast when no numbered variant
    is set)."""
    global_off = _get(percents, name + 'offset')
    per_col = {}
    for c in range(keycount):
        key = f'{name}{c}'
        if key in percents:
            per_col[c] = global_off + percents[key]
    if not per_col:
        return global_off
    return column_percents(global_off, cols, keycount, per_col)


def _perspective_dx(percents, cols, y_offset, beat_now, keycount, arrow_size):
    """The X contributions of the reprojected 3D/perspective mods: hallway
    (recede toward the vanishing point with depth) and confusiony (horizontal
    foreshortening from the Y-axis confusion tilt, incl. its offset companion
    and numbered per-column variants). Each is off when its channels are 0."""
    out = np.zeros(cols.shape[0], dtype=np.float64)
    if _get(percents, 'hallway'):
        out += hallway_x(_get(percents, 'hallway'), cols, y_offset, keycount, arrow_size)
    if (_get(percents, 'confusiony') or _get(percents, 'confusionyoffset')
            or _active(percents, 'confusiony', keycount)):
        offset = _confusion_offset(percents, 'confusiony', cols, keycount)
        out += confusiony_dx(_get(percents, 'confusiony'), cols, beat_now, keycount,
                             offset, arrow_size)
    return out


def _dx(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size):
    dx = _drunk_pair(percents, cols, y_offset, t_now, keycount, arrow_size, suffix='')
    dx += _tornado_pair(percents, cols, y_offset, keycount, arrow_size, suffix='')
    if _get(percents, 'bumpyx'):
        dx += bumpy_x(_get(percents, 'bumpyx'), y_offset,
                      _get(percents, 'bumpyxoffset'), _get(percents, 'bumpyxperiod'))
    if _get(percents, 'tanbumpyx'):
        dx += bumpy_x(_get(percents, 'tanbumpyx'), y_offset,
                      _get(percents, 'tanbumpyxoffset'),
                      _get(percents, 'tanbumpyxperiod'), is_tan=True)
    if _get(percents, 'flip'):
        dx += flip_x(_get(percents, 'flip'), cols, keycount, arrow_size)
    if _get(percents, 'invert'):
        dx += invert_x(_get(percents, 'invert'), cols, keycount, arrow_size)
    if _active(percents, 'beat', keycount):
        dx += beat_x(_per_note(percents, 'beat', cols, keycount), y_offset,
                     beat_now, _get(percents, 'beatoffset'),
                     _get(percents, 'beatperiod'), _get(percents, 'beatmult'))
    if _get(percents, 'parabolax'):
        dx += parabola(_get(percents, 'parabolax'), y_offset, arrow_size)
    if _get(percents, 'attenuatex'):
        dx += attenuate(_get(percents, 'attenuatex'), cols, y_offset, keycount, arrow_size)
    if _get(percents, 'xmode'):
        dx += xmode_x(_get(percents, 'xmode'), y_offset)
    if _active(percents, 'movex', keycount):
        dx += movex_x(_per_note(percents, 'movex', cols, keycount), arrow_size)
    dx += _perspective_dx(percents, cols, y_offset, beat_now, keycount, arrow_size)
    dx += _warp_family_sum(percents, cols, y_offset, keycount, arrow_size,
                           suffix='', tan_digital=True)
    return dx


def _warp_family_sum(percents, cols, y_offset, keycount, arrow_size,
                     suffix, tan_digital):
    """Sum the digital / zigzag / sawtooth / square / bounce warp family for
    one axis. `suffix` selects the channel set ('' for X, 'z' for Z); the X
    and Z siblings share formula shape and companion naming, so both axes call
    this. `tan_digital` adds the tandigital(z) sibling (tan kernel)."""

    def name(base):
        return base + suffix

    def per(base):
        return _per_note(percents, name(base), cols, keycount)

    def comp(base, sub):
        return _get(percents, name(base) + sub)

    out = np.zeros(cols.shape[0], dtype=np.float64)
    if _active(percents, name('digital'), keycount):
        out += digital_x(per('digital'), y_offset, comp('digital', 'offset'),
                         comp('digital', 'period'), comp('digital', 'steps'), arrow_size)
    if _active(percents, name('zigzag'), keycount):
        out += zigzag_x(per('zigzag'), y_offset, comp('zigzag', 'offset'),
                        comp('zigzag', 'period'), arrow_size)
    if _active(percents, name('sawtooth'), keycount):
        out += sawtooth_x(per('sawtooth'), y_offset, comp('sawtooth', 'period'), arrow_size)
    if _active(percents, name('square'), keycount):
        out += square_x(per('square'), y_offset, comp('square', 'offset'),
                        comp('square', 'period'), arrow_size)
    if _active(percents, name('bounce'), keycount):
        out += bounce_x(per('bounce'), y_offset, comp('bounce', 'offset'),
                        comp('bounce', 'period'), arrow_size)
    if tan_digital and _active(percents, name('tandigital'), keycount):
        out += digital_x(per('tandigital'), y_offset, comp('tandigital', 'offset'),
                         comp('tandigital', 'period'), comp('tandigital', 'steps'),
                         arrow_size, is_tan=True)
    return out


def _tipsy_dy(percents, cols, t_now, keycount, arrow_size=ARROW_SIZE):
    """The tipsy family's dy alone - the only dy component GetYPos folds
    into the visibility position (ArrowGetPercentVisible's y)."""
    dy = np.zeros(cols.shape[0], dtype=np.float64)
    if _active(percents, 'tipsy', keycount):
        dy += tipsy_y(_per_note(percents, 'tipsy', cols, keycount),
                      cols, t_now, arrow_size,
                      _get(percents, 'tipsyspeed'), _get(percents, 'tipsyoffset'))
    if _get(percents, 'tantipsy'):
        dy += tipsy_y(_get(percents, 'tantipsy'), cols, t_now, arrow_size,
                      _get(percents, 'tantipsyspeed'), _get(percents, 'tantipsyoffset'),
                      is_tan=True)
    return dy


def _dy(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size):
    dy = _tipsy_dy(percents, cols, t_now, keycount, arrow_size)
    if _get(percents, 'beaty'):
        dy += beat_y(_get(percents, 'beaty'), y_offset, beat_now,
                     _get(percents, 'beatyoffset'), _get(percents, 'beatyperiod'),
                     _get(percents, 'beatymult'))
    if _get(percents, 'parabolay'):
        dy += parabola(_get(percents, 'parabolay'), y_offset, arrow_size)
    if _get(percents, 'attenuatey'):
        dy += attenuate(_get(percents, 'attenuatey'), cols, y_offset, keycount, arrow_size)
    if _active(percents, 'movey', keycount):
        dy += movey_y(_per_note(percents, 'movey', cols, keycount), arrow_size)
    return dy


def _rotation(percents, note_beats, beat_now, n):
    rot = np.zeros(n, dtype=np.float64)
    if _get(percents, 'dizzy'):
        rot += dizzy_rotation(_get(percents, 'dizzy'), note_beats, beat_now)
    if _get(percents, 'confusion') or _get(percents, 'confusionoffset'):
        rot += confusion_rotation(_get(percents, 'confusion'), beat_now,
                                  _get(percents, 'confusionoffset'))
    return rot


def _z_push(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size):
    """Every engine +z contribution (GetZPos, ArrowEffects.cpp:1371-1538),
    accumulated in engine px for a single zoom reprojection. Covers the
    waveform warp Z siblings (digitalz / zigzagz / sawtoothz / squarez /
    bouncez + tandigitalz), bumpy/tanbumpy, drunkz/tandrunkz, tornadoz/
    tantornadoz, beatz, and attenuatez / parabolaz. Each reads its own
    `<name>z` channel + `<name>z` companions."""
    z = _warp_family_sum(percents, cols, y_offset, keycount, arrow_size,
                         suffix='z', tan_digital=True)
    if _get(percents, 'bumpy'):
        z += bumpy_z(_get(percents, 'bumpy'), y_offset,
                     _get(percents, 'bumpyoffset'), _get(percents, 'bumpyperiod'))
    if _get(percents, 'tanbumpy'):
        z += bumpy_z(_get(percents, 'tanbumpy'), y_offset,
                     _get(percents, 'tanbumpyoffset'), _get(percents, 'tanbumpyperiod'),
                     is_tan=True)
    z += _drunk_pair(percents, cols, y_offset, t_now, keycount, arrow_size, suffix='z')
    z += _tornado_pair(percents, cols, y_offset, keycount, arrow_size, suffix='z')
    if _get(percents, 'beatz'):
        z += beat_z(_get(percents, 'beatz'), y_offset, beat_now,
                    _get(percents, 'beatzoffset'), _get(percents, 'beatzperiod'),
                    _get(percents, 'beatzmult'))
    if _get(percents, 'parabolaz'):
        z += parabola(_get(percents, 'parabolaz'), y_offset, arrow_size)
    if _get(percents, 'attenuatez'):
        z += attenuate(_get(percents, 'attenuatez'), cols, y_offset, keycount, arrow_size)
    return z


def _zoom(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size, n,
          z_push=None):
    base = zoom_from_mini(_get(percents, 'mini')) * tiny_zoom(_get(percents, 'tiny'))
    zoom = np.full(n, base, dtype=np.float64)

    pulse = pulse_zoom(_get(percents, 'pulseinner'), _get(percents, 'pulseouter'),
                       y_offset, _get(percents, 'pulseoffset'),
                       _get(percents, 'pulseperiod'), arrow_size)
    zoom = zoom * pulse

    shrink_mult, shrink_add = shrink_zoom(_get(percents, 'shrinkmult'),
                                          _get(percents, 'shrinklinear'),
                                          y_offset, arrow_size)
    zoom = zoom * shrink_mult + shrink_add

    # The engine +z push scales a note by the perspective divide. The
    # projected note path applies that through the camera (real depth),
    # so it passes z_push=<the array> and we DON'T also fake it as zoom.
    # `z_push=None` is the 2D fallback: reproject to zoom as before.
    if z_push is None:
        z_push = _z_push(percents, cols, y_offset, t_now, beat_now,
                         keycount, arrow_size)
        if np.any(z_push):
            zoom = zoom * waveform_z_zoom(z_push)

    if (_get(percents, 'confusionx') or _get(percents, 'confusionxoffset')
            or _active(percents, 'confusionx', keycount)):
        offset = _confusion_offset(percents, 'confusionx', cols, keycount)
        zoom = zoom * confusionx_zoom(_get(percents, 'confusionx'), beat_now, offset)
    return zoom


def _note_tilt(percents, y_offset, n):
    """(rot_x, rot_y) per-note out-of-plane tilt in degrees: roll ->
    RotationX, twirl -> RotationY (ArrowEffects GetRotationX/Y = effect *
    yOffset/2). These need the projected note path; a flat draw drops
    them (a 2D sprite cannot tilt out of plane). Rest 0 -> no tilt."""
    roll = _get(percents, 'roll')
    twirl = _get(percents, 'twirl')
    rot_x = roll * y_offset / 2.0 if roll else np.zeros(n)
    rot_y = twirl * y_offset / 2.0 if twirl else np.zeros(n)
    return (np.broadcast_to(rot_x, (n,)).astype(np.float64),
            np.broadcast_to(rot_y, (n,)).astype(np.float64))


def _alpha(percents, cols, y_pos, t_now):
    vis = dict(percents)
    vis['_blink_adjust'] = blink_adjust(_get(percents, 'blink'), t_now)
    alpha = alpha_from_visible(percent_visible(vis, cols, y_pos))
    if _get(percents, 'boomerang'):
        alpha = alpha * boomerang_visibility(_get(percents, 'boomerang'), y_pos)
    return alpha


def note_offsets(percents: dict, cols: np.ndarray, y_offset: np.ndarray,
                 t_now: float, beat_now: float, keycount: int,
                 note_beats: np.ndarray | None = None,
                 bps: float = 2.0, arrow_size: float = ARROW_SIZE,
                 rand_seed: int = 0, project_3d: bool = False) -> NoteOffsets:
    """Sum every implemented note-position mod into a `NoteOffsets`.

    `percents` maps mod name -> value (fraction); numbered per-column
    variants (`drunk0`, `movex1`, ...) are picked up automatically.
    `cols` is an int array of column indices; `y_offset` the pre-mod
    signed scroll offset per note; `note_beats` the per-note beat (for
    dizzy; falls back to beat_now if omitted). `bps` is retained for caller
    back-compat but no longer read (ITGmania's beat uses the beatmult
    companion, not a bpm-derived divisor - see `beat_factor`).

    Covers: the drunk / tipsy / tornado / bumpy / beat position mods and
    their offset/period/speed/mult companions; the tan* kernels (tandrunk,
    tantipsy, tantornado, tanbumpy(x), tandigital + z); the digital /
    waveform warp family (digital, zigzag, sawtooth, square, bounce + *z);
    beaty/beatz; attenuate x/y/z + parabola x/y/z; pulse and shrink zoom;
    and boomerang's visibility fold (its position parabola lives in
    `accel_y_offset`). Every companion rides the same `percents` dict.

    The out-of-plane confusion tilts confusionx / confusiony (ReceptorGet-
    RotationX/Y, with their *offset companions and numbered per-column
    variants) reproject into the 2D pipeline: confusionx as a uniform
    zoom (vertical foreshortening, `confusionx_zoom`), confusiony as a
    per-column dx (horizontal foreshortening, `confusiony_dx`). The
    SCALAR variants normally never reach here - the NotITG field
    projection renders them as true perspective and the consumer zeroes
    them (module header, "confusion") - so these kernels serve the
    per-column variants and the projection's fallback cases. `hallway`
    (a notefield perspective mod, no ArrowEffects formula) is ported as a
    per-note dx that recedes columns toward the vanishing point with depth
    (`hallway_x`).

    DEFERRED (2D limitation, documented in the module header): roll
    (RotationX) and twirl (RotationY) are out-of-plane tilts a 2D sprite
    can't express, so they contribute nothing here (confusionx/y differ:
    they are constant tilts, faithfully reprojectable as foreshortening,
    while roll/twirl scale their tilt with y_offset per note). `grain` /
    `granulate` (hold step-size) and dizzyholds (hold-render-specific) stay
    deferred; sawtooth's offset companion is unread in the ported engine
    formula (see `sawtooth_x`)."""
    cols = np.asarray(cols)
    y_offset = np.asarray(y_offset, dtype=np.float64)
    n = cols.shape[0]
    if note_beats is None:
        note_beats = np.full(n, beat_now, dtype=np.float64)
    else:
        note_beats = np.asarray(note_beats, dtype=np.float64)

    dx = _dx(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size)
    dy = _dy(percents, cols, y_offset, t_now, beat_now, keycount, arrow_size)
    rotation = _rotation(percents, note_beats, beat_now, n)
    # The projected note path takes real per-note depth (z) + out-of-
    # plane tilt (roll/twirl) and lets the camera do the perspective; the
    # 2D path reprojects z to zoom and drops the tilts (a flat sprite).
    z = rot_x = rot_y = None
    if project_3d:
        z = _z_push(percents, cols, y_offset, t_now, beat_now, keycount,
                    arrow_size)
        z = np.broadcast_to(z, (n,)).astype(np.float64)
        rot_x, rot_y = _note_tilt(percents, y_offset, n)
    zoom = _zoom(percents, cols, y_offset, t_now, beat_now, keycount,
                 arrow_size, n, z_push=z)
    # Visibility samples GetYPos(..., WithReverse=false): the raw scroll
    # offset plus TIPSY ONLY (ArrowEffects.cpp:441-444/159-176) - the
    # other dy mods (beaty/movey/parabola...) displace the drawn note
    # but never move it through the hidden/sudden windows.
    alpha = _alpha(percents, cols, y_offset + _tipsy_dy(percents, cols, t_now,
                                                        keycount, arrow_size),
                   t_now)

    return NoteOffsets(dx=dx, dy=dy, rotation_deg=rotation,
                       alpha_mult=alpha, zoom=zoom, z=z, rot_x=rot_x,
                       rot_y=rot_y)


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
