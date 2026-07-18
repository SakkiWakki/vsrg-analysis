"""A slow, scalar, deliberately literal port of ITGmania's ArrowEffects.

This is the ORACLE for the differential parity tests in
`tests/test_engine_parity.py`. It exists to confirm the vectorized
production port (`analysis/player/render/mods/arrow_effects.py`) against
the actual engine, on the COMPOSED pipeline rather than one formula at a
time.

# Source and fork

Authoritative source: ITGmania `src/ArrowEffects.cpp` (the open fork that
merged the NotITG-era mod set - waveform warp family, tan/cosec siblings,
per-axis beat, attenuate/parabola, pulse/shrink, confusion X/Y). The line
references in the comments below are that file's numbering. Where OpenITG
diverges (it lacks the whole NotITG extension set) the ITGmania form is
followed and the divergence noted inline. PlayerOptions::
GetReversePercentForColumn is ported from ITGmania
`src/PlayerOptions.cpp:1681`.

This module is a TRANSLITERATION: scalar, one column and one note at a
time, no numpy vectorization, no cleverness. Every function mirrors an
engine function; the point is that a bug in the fast port shows up as a
divergence from this line-faithful one, not that this one is fast.

# Engine constants (ITGmania Themes/_fallback/metrics.ini [ArrowEffects])

Baked from the fallback theme - the values the formulas were tuned
against and the engine's out-of-the-box behavior. The engine reads them
per-theme through ThemeMetric; a modchart player has one theme, so the
fallback set is the ground truth.

# Deliberate substitutions (these are the engine's env, made explicit)

- GetTime(): the engine's default ModTimerType reads
  RageTimer::GetTimeSinceStart() - a WALL CLOCK. A replay needs a
  deterministic, scrub-exact clock, so the whole comparison feeds song
  time as `t_now` to BOTH sides. This reference therefore also takes
  `t_now` as its time; it is faithful to ModTimerType_Song, which the
  engine does support, and is the one intentional-substitution axis
  (harness class iii).
- m_fSongBeatVisible: the beat the engine feeds confusion / dizzy / beat.
  The harness feeds the same `beat_now` to both sides.
- m_NotefieldZoom / field_zoom = 1.0 (no ScreenGameplay zoom in a bare
  field); m_fPerspectiveTilt = 0 so GetNoteFieldHeight() == SCREEN_HEIGHT.
- Cosecant flag off unless a `_cosecant` percent is passed (the tan family
  uses tan(), not 1/sin()).
"""
from __future__ import annotations

import math

ARROW_SIZE = 64.0
SCREEN_WIDTH = 640.0
SCREEN_HEIGHT = 480.0
PI = math.pi

# LoadMenuPerspective camera distance (fov 45 over the design width),
# derived independently of production: the documented z->zoom contract is
# the center-plane perspective scale d / (d - z), inverted by
# z = d * (1 - 1/zoom).
EYE_DISTANCE = (SCREEN_WIDTH / 2.0) / math.tan(math.radians(45.0) / 2.0)


def z_zoom(z):
    """The z->zoom contract: center-plane perspective scale of a +z push."""
    return EYE_DISTANCE / (EYE_DISTANCE - z)


def zoom_to_z(zoom):
    """Exact inverse of `z_zoom`, recovering the summed z from a zoom."""
    return EYE_DISTANCE * (1.0 - 1.0 / zoom)

# metrics.ini [ArrowEffects] fallback defaults.
BLINK_MOD_FREQUENCY = 0.3333
BOOST_MOD_MIN_CLAMP = -400.0
BOOST_MOD_MAX_CLAMP = 400.0
BRAKE_MOD_MIN_CLAMP = -400.0
BRAKE_MOD_MAX_CLAMP = 400.0
WAVE_MOD_MAGNITUDE = 20.0
WAVE_MOD_HEIGHT = 38.0
BOOMERANG_PEAK_PERCENTAGE = 0.75
EXPAND_MULTIPLIER_FREQUENCY = 3.0
EXPAND_MULTIPLIER_SCALE_FROM_LOW = -1.0
EXPAND_MULTIPLIER_SCALE_FROM_HIGH = 1.0
EXPAND_MULTIPLIER_SCALE_TO_LOW = 0.75
EXPAND_MULTIPLIER_SCALE_TO_HIGH = 1.75
EXPAND_SPEED_SCALE_FROM_LOW = 0.0
EXPAND_SPEED_SCALE_FROM_HIGH = 1.0
EXPAND_SPEED_SCALE_TO_LOW = 1.0
TIPSY_TIMER_FREQUENCY = 1.2
TIPSY_COLUMN_FREQUENCY = 1.8
TIPSY_ARROW_MAGNITUDE = 0.4
TORNADO_POSITION_SCALE_TO_LOW = -1.0
TORNADO_POSITION_SCALE_TO_HIGH = 1.0
TORNADO_OFFSET_FREQUENCY = 6.0
TORNADO_OFFSET_SCALE_FROM_LOW = -1.0
TORNADO_OFFSET_SCALE_FROM_HIGH = 1.0
DRUNK_COLUMN_FREQUENCY = 0.2
DRUNK_OFFSET_FREQUENCY = 10.0
DRUNK_ARROW_MAGNITUDE = 0.5
BEAT_OFFSET_HEIGHT = 15.0
BEAT_PI_HEIGHT = 2.0
TINY_PERCENT_BASE = 0.5
TINY_PERCENT_GATE = 1.0
CENTER_LINE_Y = 160.0
FADE_DIST_Y = 40.0
# metrics.ini does not ship Beat[YZ]OffsetHeight / Beat[YZ]PIHeight or the
# Drunk[Z] frequency metrics in the fallback theme this reference was read
# from; ITGmania defaults the Y/Z beat metrics to the same 15 / 2 as the X
# beat, and DrunkZ to the same 0.2 / 10 / 0.5 as drunk. The production port
# makes the same assumption, so the two agree by construction; flagged as a
# reference-ambiguity, not a verified engine value.
BEAT_Y_OFFSET_HEIGHT = 15.0
BEAT_Y_PI_HEIGHT = 2.0
BEAT_Z_OFFSET_HEIGHT = 15.0
BEAT_Z_PI_HEIGHT = 2.0
DRUNK_Z_COLUMN_FREQUENCY = 0.2
DRUNK_Z_OFFSET_FREQUENCY = 10.0
DRUNK_Z_ARROW_MAGNITUDE = 0.5


def _scale(x, l1, h1, l2, h2):
    """SCALE macro (RageUtil.h:62)."""
    return (x - l1) * (h2 - l2) / (h1 - l1) + l2


def _clamp(x, lo, hi):
    """rage_clamp: clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def _quantize(f, interval):
    """Quantize (RageUtil.h:175): int((f + i/2)/i)*i. The int() cast
    TRUNCATES toward zero (C semantics), which differs from floor() for
    negative arguments - load-bearing for blink, whose sine goes negative."""
    return int((f + interval / 2.0) / interval) * interval


def _select_tan(angle, cosecant):
    """SelectTanType (ArrowEffects.cpp:209)."""
    if cosecant:
        return 1.0 / math.sin(angle)
    return math.tan(angle)


def rage_triangle(angle):
    """RageTriangle (RageMath.cpp:593). fmod can go negative for a negative
    angle, so the engine adds 2*PI before the piecewise selection."""
    a = math.fmod(angle, 2.0 * PI)
    if a < 0.0:
        a += 2.0 * PI
    result = a * (1.0 / PI)
    if result < 0.5:
        return result * 2.0
    if result < 1.5:
        return 1.0 - (result - 0.5) * 2.0
    return -4.0 + result * 2.0


def rage_square(angle):
    """RageSquare (RageMath.cpp:584). The <0.01 guard nudges a near-zero
    angle up by 2*PI (a hold-flicker hack); for a small POSITIVE angle in
    [0, 0.01) that flips the result from +1 to -1, so it is NOT a no-op."""
    a = math.fmod(angle, 2.0 * PI)
    if a < 0.01:
        a += 2.0 * PI
    return -1.0 if a >= PI else 1.0


def column_x_offset(col, keycount):
    """pStyle->m_ColumnInfo[col].fXOffset for a plain evenly-spaced N-key
    style: (col - (N-1)/2) * ARROW_SIZE."""
    return (col - (keycount - 1) / 2.0) * ARROW_SIZE


# --- accel family: the y_offset reshapers (GetYOffset, :540-704) ----------

def get_note_field_height():
    """GetNoteFieldHeight (:145): SCREEN_HEIGHT + |tilt|*200; tilt 0 here."""
    return SCREEN_HEIGHT


def get_y_offset_accel(percents, y_offset):
    """The accel section of GetYOffset (:594-703) applied to a raw
    y_offset. Returns the reshaped y_offset (scroll speed is 1.0 here, so
    the trailing `*= fScrollSpeed` is identity).

    Faithful to the engine's ORDER and its y<0 early-out: notes past the
    receptor (y<0) are returned untouched (`:595` `if (fYOffset < 0)
    return fYOffset * fScrollSpeed;`), BEFORE boost/brake/wave/parabola_y
    are even considered. boomerang folds the whole range after."""
    boost = percents.get('boost', 0.0)
    brake = percents.get('brake', 0.0)
    wave = percents.get('wave', 0.0)
    wave_period = percents.get('waveperiod', 0.0)
    parabola_y = percents.get('parabolay', 0.0)
    expand = percents.get('expand', 0.0)
    boomerang = percents.get('boomerang', 0.0)

    y = float(y_offset)
    if y < 0.0:
        return y

    y_adjust = 0.0
    if boost != 0.0:
        eff_h = get_note_field_height()
        new_y = y * 1.5 / ((y + eff_h / 1.2) / eff_h)
        adj = boost * (new_y - y)
        y_adjust += _clamp(adj, BOOST_MOD_MIN_CLAMP, BOOST_MOD_MAX_CLAMP)
    if brake != 0.0:
        eff_h = get_note_field_height()
        scale = _scale(y, 0.0, eff_h, 0.0, 1.0)
        new_y = y * scale
        adj = brake * (new_y - y)
        y_adjust += _clamp(adj, BRAKE_MOD_MIN_CLAMP, BRAKE_MOD_MAX_CLAMP)
    if wave != 0.0:
        y_adjust += wave * WAVE_MOD_MAGNITUDE * math.sin(
            y / ((wave_period * WAVE_MOD_HEIGHT) + WAVE_MOD_HEIGHT))
    # NOTE: parabola_y is read HERE in GetYOffset (:638), not in GetYPos.
    # It reshapes the position, unlike attenuate_y / beat_y which are dy in
    # GetYPos. The production port routes parabolay through _dy (GetYPos),
    # which is the same additive contribution but bypasses the y<0 early-out
    # and the boomerang fold. Kept faithful here; flagged in the harness.
    if parabola_y != 0.0:
        y_adjust += parabola_y * (y / ARROW_SIZE) * (y / ARROW_SIZE)

    y += y_adjust

    if boomerang != 0.0:
        y = (-1.0 * y * y / SCREEN_HEIGHT) + 1.5 * y

    if expand != 0.0:
        # WALLCLOCK -> song time: the engine accumulates m_fExpandSeconds
        # from real dt; we key the cos phase to song time via _expand_phase
        # (radians) the same way the production port does, then apply the
        # engine's SCALE-of-cos multiplier chain.
        phase = percents.get('_expand_phase', 0.0)
        expand_mult = _scale(
            math.cos(phase), EXPAND_MULTIPLIER_SCALE_FROM_LOW,
            EXPAND_MULTIPLIER_SCALE_FROM_HIGH, EXPAND_MULTIPLIER_SCALE_TO_LOW,
            EXPAND_MULTIPLIER_SCALE_TO_HIGH)
        scroll = _scale(expand, EXPAND_SPEED_SCALE_FROM_LOW,
                        EXPAND_SPEED_SCALE_FROM_HIGH, EXPAND_SPEED_SCALE_TO_LOW,
                        expand_mult)
        y *= scroll
    return y


def boomerang_peak():
    """fPeakYOffsetOut (:650): the folded offset at raw peak p=H*0.75."""
    p = SCREEN_HEIGHT * BOOMERANG_PEAK_PERCENTAGE
    return (-1.0 * p * p / SCREEN_HEIGHT) + 1.5 * p


# --- reverse (PlayerOptions::GetReversePercentForColumn, :1681) ------------

def get_reverse_percent_for_column(percents, col, keycount):
    """PlayerOptions::GetReversePercentForColumn (PlayerOptions.cpp:1681).

    reverse (all cols) + per-column m_fReverse[col] (numbered `reverse<c>`)
    + split (col >= N/2) + alternate (odd col) + cross (middle half). Wraps
    >2 mod 2, then (1,2] mirrors back down via SCALE(f,1,2,1,0)."""
    f = percents.get('reverse', 0.0)
    f += percents.get(f'reverse{col}', 0.0)
    if col >= keycount // 2:
        f += percents.get('split', 0.0)
    if (col % 2) == 1:
        f += percents.get('alternate', 0.0)
    first_cross = keycount // 4
    last_cross = keycount - 1 - first_cross
    if first_cross <= col <= last_cross:
        f += percents.get('cross', 0.0)
    if f > 2.0:
        f = math.fmod(f, 2.0)
    if f > 1.0:
        f = _scale(f, 1.0, 2.0, 1.0, 0.0)
    return f


def reverse_shift_and_scale(percents, col, keycount, y_reverse_offset):
    """ArrowGetReverseShiftAndScale (:706). The scale flips the note's
    distance from the receptor; the shift moves the receptor along the
    mirrored path; centered pulls it back to field center. mini scales the
    reverse shift by the field zoom (1 - mini*0.5)."""
    mini = percents.get('mini', 0.0)
    zoom = 1.0 - mini * 0.5
    if abs(zoom) < 0.01:
        zoom = 0.01
    r = get_reverse_percent_for_column(percents, col, keycount)
    shift = _scale(r, 0.0, 1.0, -y_reverse_offset / zoom / 2.0,
                   y_reverse_offset / zoom / 2.0)
    centered = percents.get('centered', 0.0)
    shift = _scale(centered, 0.0, 1.0, shift, 0.0)
    scale = _scale(r, 0.0, 1.0, 1.0, -1.0)
    return shift, scale


# --- GetYPos (:728): reverse, then tipsy/attenuate_y/beat_y ----------------

def get_y_pos(percents, col, keycount, y_offset, t_now, beat_now,
              y_reverse_offset, with_reverse=True):
    """ArrowEffects::GetYPos (:728). f starts at the (accel-adjusted)
    y_offset, is optionally reversed, then gets tipsy / attenuate_y /
    beat_y added. QUANTIZE_ARROW_Y is false in fallback, so no floor."""
    f = float(y_offset)
    if with_reverse:
        shift, scale = reverse_shift_and_scale(
            percents, col, keycount, y_reverse_offset)
        f *= scale
        f += shift

    f += percents.get('tipsy', 0.0) * _tipsy_result(percents, col, t_now, False)
    f += percents.get('tantipsy', 0.0) * _tipsy_result(percents, col, t_now, True)

    att_y = percents.get('attenuatey', 0.0)
    if att_y != 0.0:
        xoff = column_x_offset(col, keycount)
        f += att_y * (y_offset / ARROW_SIZE) * (y_offset / ARROW_SIZE) * \
            (xoff / ARROW_SIZE)

    beaty = percents.get('beaty', 0.0)
    if beaty != 0.0:
        factor = _beat_factor(percents, beat_now, 'beaty')
        shift = factor * math.sin(
            y_offset / ((percents.get('beatyperiod', 0.0) * BEAT_Y_OFFSET_HEIGHT)
                        + BEAT_Y_OFFSET_HEIGHT) + PI / BEAT_Y_PI_HEIGHT)
        f += beaty * shift
    return f


def _tipsy_result(percents, col, t_now, is_tan):
    """UpdateTipsy per-column result (:300). speed scales the timer term,
    offset scales the per-column term. cos, or tan/cosec for the tan sib."""
    prefix = 'tantipsy' if is_tan else 'tipsy'
    speed = percents.get(prefix + 'speed', 0.0)
    offset = percents.get(prefix + 'offset', 0.0)
    time = t_now
    angle = (time * ((speed * TIPSY_TIMER_FREQUENCY) + TIPSY_TIMER_FREQUENCY)
             + col * ((offset * TIPSY_COLUMN_FREQUENCY) + TIPSY_COLUMN_FREQUENCY))
    kernel = (_select_tan(angle, percents.get('_cosecant', False))
              if is_tan else math.cos(angle))
    return kernel * ARROW_SIZE * TIPSY_ARROW_MAGNITUDE


# --- beat factor (UpdateBeat, :260) ---------------------------------------

def _beat_factor(percents, beat_now, base):
    """UpdateBeat (:260): the once-per-beat pulse * 20, per axis. `base`
    selects the offset/mult companions ('beat'/'beaty'/'beatz')."""
    offset = percents.get(base + 'offset', 0.0)
    mult = percents.get(base + 'mult', 0.0)
    accel_time, total_time = 0.2, 0.5
    beat = (beat_now + accel_time + offset) * (mult + 1.0)
    even = (int(beat) % 2) != 0
    if beat < 0.0:
        return 0.0
    beat -= math.trunc(beat)
    beat += 1.0
    beat -= math.trunc(beat)
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


# --- tornado (CalculateTornadoOffsetFromMagnitude, :217) ------------------

def _tornado_window(col, keycount, dimension):
    """The [min, max] x-offset window over cols [col-w, col+w]. Width is 2
    for x (dim 0) in a wide field (>4 cols), else 3 (:355-378)."""
    wide = keycount > 4
    width = 2 if (dimension == 0 and wide) else 3
    start = _clamp(col - width, 0, keycount - 1)
    end = _clamp(col + width, 0, keycount - 1)
    offsets = [column_x_offset(i, keycount) for i in range(start, end + 1)]
    return min(offsets), max(offsets)


def _tornado_offset(percents, col, keycount, y_offset, base, dimension, is_tan):
    """CalculateTornadoOffsetFromMagnitude (:217). field_zoom = 1.0."""
    magnitude = percents.get(base, 0.0)
    effect_offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    real = column_x_offset(col, keycount)
    min_t, max_t = _tornado_window(col, keycount, dimension)
    position_between = _scale(real, min_t, max_t,
                              TORNADO_POSITION_SCALE_TO_LOW,
                              TORNADO_POSITION_SCALE_TO_HIGH)
    rads = math.acos(position_between)
    freq = TORNADO_OFFSET_FREQUENCY
    rads += (y_offset + effect_offset) * ((period * freq) + freq) / SCREEN_HEIGHT
    processed = (_select_tan(rads, percents.get('_cosecant', False))
                 if is_tan else math.cos(rads))
    adjusted = _scale(processed, TORNADO_OFFSET_SCALE_FROM_LOW,
                      TORNADO_OFFSET_SCALE_FROM_HIGH, min_t, max_t)
    return (adjusted - real) * magnitude


def _drunk_offset(percents, col, keycount, y_offset, t_now, base, is_tan,
                  col_freq, offset_freq, magnitude_const):
    """The drunk / drunkz kernel (CalculateDrunkAngle, :242 + GetXPos drunk
    :854). speed scales time, offset scales col term, period scales yoff."""
    magnitude = percents.get(base, 0.0)
    speed = percents.get(base + 'speed', 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    angle = (t_now * (1.0 + speed)
             + col * ((offset * col_freq) + col_freq)
             + y_offset * ((period * offset_freq) + offset_freq) / SCREEN_HEIGHT)
    kernel = (_select_tan(angle, percents.get('_cosecant', False))
              if is_tan else math.cos(angle))
    return magnitude * (kernel * ARROW_SIZE * magnitude_const)


def _bumpy_kernel(percents, y_offset, base, is_tan):
    """CalculateBumpyAngle (:251) * 40 * sin (or tan/cosec)."""
    magnitude = percents.get(base, 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    angle = (y_offset + 100.0 * offset) / ((period * 16.0) + 16.0)
    kernel = (_select_tan(angle, percents.get('_cosecant', False))
              if is_tan else math.sin(angle))
    return magnitude * 40.0 * kernel


def _digital_kernel(percents, y_offset, base, is_tan):
    """digital / tandigital (:943): round-quantized sine, steps+1 levels."""
    magnitude = percents.get(base, 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    steps = percents.get(base + 'steps', 0.0)
    angle = PI * (y_offset + 1.0 * offset) / (ARROW_SIZE + period * ARROW_SIZE)
    kernel = (_select_tan(angle, percents.get('_cosecant', False))
              if is_tan else math.sin(angle))
    return (magnitude * ARROW_SIZE * 0.5) * round_half_away(
        (steps + 1) * kernel) / (steps + 1)


def round_half_away(x):
    """std::round: round-half-away-from-zero (differs from numpy's
    round-half-to-even at the .5 tie, load-bearing for digital)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def _zigzag_kernel(percents, y_offset, base):
    """zigzag (:908): triangle wave, +/- ARROW_SIZE/2."""
    magnitude = percents.get(base, 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    result = rage_triangle(
        PI * (1.0 / (period + 1.0)) * ((y_offset + 100.0 * offset) / ARROW_SIZE))
    return (magnitude * ARROW_SIZE / 2.0) * result


def _sawtooth_kernel(percents, y_offset, base):
    """sawtooth (:919): fractional-part ramp, scaled to ARROW_SIZE. No
    offset companion in the engine formula."""
    magnitude = percents.get(base, 0.0)
    period = percents.get(base + 'period', 0.0)
    ramp = (0.5 / (period + 1.0) * y_offset) / ARROW_SIZE
    return (magnitude * ARROW_SIZE) * (ramp - math.floor(ramp))


def _square_kernel(percents, y_offset, base):
    """square (:968): square wave, +/- ARROW_SIZE/2."""
    magnitude = percents.get(base, 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    result = rage_square(
        PI * (y_offset + 1.0 * offset) / (ARROW_SIZE + period * ARROW_SIZE))
    return (magnitude * ARROW_SIZE * 0.5) * result


def _bounce_kernel(percents, y_offset, base):
    """bounce (:979): |sin|, base period 60, scaled to ARROW_SIZE/2."""
    magnitude = percents.get(base, 0.0)
    offset = percents.get(base + 'offset', 0.0)
    period = percents.get(base + 'period', 0.0)
    amt = abs(math.sin((y_offset + 1.0 * offset) / (60.0 + period * 60.0)))
    return magnitude * ARROW_SIZE * 0.5 * amt


# --- GetXPos (:807) --------------------------------------------------------

def get_x_pos(percents, col, keycount, y_offset):
    """ArrowEffects::GetXPos (:807), in the engine's summation order. The
    column's own fXOffset is added mid-stream (:1022), then tiny multiplies
    the WHOLE accumulated offset (:1025). Returns the pixel offset from
    field center (so the note's screen x = field_center + this)."""
    off = 0.0
    if percents.get('tornado', 0.0) != 0.0:
        off += _tornado_offset(percents, col, keycount, y_offset, 'tornado', 0, False)
    if percents.get('tantornado', 0.0) != 0.0:
        off += _tornado_offset(percents, col, keycount, y_offset, 'tantornado', 0, True)
    if percents.get('bumpyx', 0.0) != 0.0:
        off += _bumpy_kernel(percents, y_offset, 'bumpyx', False)
    if percents.get('tanbumpyx', 0.0) != 0.0:
        off += _bumpy_kernel(percents, y_offset, 'tanbumpyx', True)
    if percents.get('drunk', 0.0) != 0.0:
        off += _drunk_offset(percents, col, keycount, y_offset, _time(percents),
                             'drunk', False, DRUNK_COLUMN_FREQUENCY,
                             DRUNK_OFFSET_FREQUENCY, DRUNK_ARROW_MAGNITUDE)
    if percents.get('tandrunk', 0.0) != 0.0:
        off += _drunk_offset(percents, col, keycount, y_offset, _time(percents),
                             'tandrunk', True, DRUNK_COLUMN_FREQUENCY,
                             DRUNK_OFFSET_FREQUENCY, DRUNK_ARROW_MAGNITUDE)
    if percents.get('flip', 0.0) != 0.0:
        new_col = int(round(_scale(col, 0, keycount - 1, keycount - 1, 0)))
        dist = column_x_offset(new_col, keycount) - column_x_offset(col, keycount)
        off += dist * percents['flip']
    if percents.get('invert', 0.0) != 0.0:
        off += _invert_distance(col, keycount) * percents['invert']
    if percents.get('beat', 0.0) != 0.0:
        factor = _beat_factor(percents, _beat(percents), 'beat')
        shift = factor * math.sin(
            y_offset / ((percents.get('beatperiod', 0.0) * BEAT_OFFSET_HEIGHT)
                        + BEAT_OFFSET_HEIGHT) + PI / BEAT_PI_HEIGHT)
        off += percents['beat'] * shift
    if percents.get('zigzag', 0.0) != 0.0:
        off += _zigzag_kernel(percents, y_offset, 'zigzag')
    if percents.get('sawtooth', 0.0) != 0.0:
        off += _sawtooth_kernel(percents, y_offset, 'sawtooth')
    if percents.get('parabolax', 0.0) != 0.0:
        off += percents['parabolax'] * (y_offset / ARROW_SIZE) * (y_offset / ARROW_SIZE)
    if percents.get('attenuatex', 0.0) != 0.0:
        xoff = column_x_offset(col, keycount)
        off += percents['attenuatex'] * (y_offset / ARROW_SIZE) * \
            (y_offset / ARROW_SIZE) * (xoff / ARROW_SIZE)
    if percents.get('digital', 0.0) != 0.0:
        off += _digital_kernel(percents, y_offset, 'digital', False)
    if percents.get('tandigital', 0.0) != 0.0:
        off += _digital_kernel(percents, y_offset, 'tandigital', True)
    if percents.get('square', 0.0) != 0.0:
        off += _square_kernel(percents, y_offset, 'square')
    if percents.get('bounce', 0.0) != 0.0:
        off += _bounce_kernel(percents, y_offset, 'bounce')
    if percents.get('xmode', 0.0) != 0.0:
        # single-side field: + yOffset (P2/right-half sign split not modeled).
        off += percents['xmode'] * y_offset

    off += column_x_offset(col, keycount)

    if percents.get('movex', 0.0) != 0.0 or percents.get(f'movex{col}', 0.0) != 0.0:
        # GetMoveX (:1165) is added by NoteDisplay AFTER GetXPos, not inside
        # it; it is a per-column translate of 64*movex. Folded here so the
        # reference's total x matches the production port (which sums movex
        # into dx). Flagged in the harness: engine applies it post-tiny.
        off += ARROW_SIZE * percents.get(f'movex{col}', percents.get('movex', 0.0))

    if percents.get('tiny', 0.0) != 0.0:
        tiny = percents['tiny']
        tiny = min(math.pow(TINY_PERCENT_BASE, tiny), TINY_PERCENT_GATE)
        off *= tiny
    return off


def _invert_distance(col, keycount):
    """m_fInvertDistance[col] (Update, :429): mirror within each half of a
    single-side field. iNumColsPerSide = keycount (one side)."""
    per_side = keycount
    on_side = col % per_side
    left_of_mid = (per_side - 1) // 2
    right_of_mid = (per_side + 1) // 2
    if on_side <= left_of_mid:
        first, last = 0, left_of_mid
    elif on_side >= right_of_mid:
        first, last = right_of_mid, per_side - 1
    else:
        first = last = on_side // 2
    new_on_side = 0 if first == last \
        else int(round(_scale(on_side, first, last, last, first)))
    new_col = new_on_side  # iSideIndex 0 for single side
    return column_x_offset(new_col, keycount) - column_x_offset(col, keycount)


# --- GetZPos (:1371) -------------------------------------------------------

def get_z_pos(percents, col, keycount, y_offset, t_now, beat_now):
    """ArrowEffects::GetZPos (:1371): every +z push in engine px, summed."""
    z = 0.0
    if percents.get('tornadoz', 0.0) != 0.0:
        z += _tornado_offset(percents, col, keycount, y_offset, 'tornadoz', 2, False)
    if percents.get('tantornadoz', 0.0) != 0.0:
        z += _tornado_offset(percents, col, keycount, y_offset, 'tantornadoz', 2, True)
    if percents.get('bumpy', 0.0) != 0.0:
        z += _bumpy_kernel(percents, y_offset, 'bumpy', False)
    if percents.get(f'bumpy{col}', 0.0) != 0.0:
        # m_fBumpy[col] per-column (:1406): shares the global bumpy angle.
        offset = percents.get('bumpyoffset', 0.0)
        period = percents.get('bumpyperiod', 0.0)
        angle = (y_offset + 100.0 * offset) / ((period * 16.0) + 16.0)
        z += percents[f'bumpy{col}'] * 40.0 * math.sin(angle)
    if percents.get('tanbumpy', 0.0) != 0.0:
        z += _bumpy_kernel(percents, y_offset, 'tanbumpy', True)
    if percents.get('zigzagz', 0.0) != 0.0:
        z += _zigzag_kernel(percents, y_offset, 'zigzagz')
    if percents.get('sawtoothz', 0.0) != 0.0:
        z += _sawtooth_kernel(percents, y_offset, 'sawtoothz')
    if percents.get('parabolaz', 0.0) != 0.0:
        z += percents['parabolaz'] * (y_offset / ARROW_SIZE) * (y_offset / ARROW_SIZE)
    if percents.get('attenuatez', 0.0) != 0.0:
        xoff = column_x_offset(col, keycount)
        z += percents['attenuatez'] * (y_offset / ARROW_SIZE) * \
            (y_offset / ARROW_SIZE) * (xoff / ARROW_SIZE)
    if percents.get('drunkz', 0.0) != 0.0:
        z += _drunk_offset(percents, col, keycount, y_offset, t_now, 'drunkz',
                           False, DRUNK_Z_COLUMN_FREQUENCY,
                           DRUNK_Z_OFFSET_FREQUENCY, DRUNK_Z_ARROW_MAGNITUDE)
    if percents.get('tandrunkz', 0.0) != 0.0:
        z += _drunk_offset(percents, col, keycount, y_offset, t_now, 'tandrunkz',
                           True, DRUNK_Z_COLUMN_FREQUENCY, DRUNK_Z_OFFSET_FREQUENCY,
                           DRUNK_Z_ARROW_MAGNITUDE)
    if percents.get('beatz', 0.0) != 0.0:
        factor = _beat_factor(percents, beat_now, 'beatz')
        shift = factor * math.sin(
            y_offset / ((percents.get('beatzperiod', 0.0) * BEAT_Z_OFFSET_HEIGHT)
                        + BEAT_Z_OFFSET_HEIGHT) + PI / BEAT_Z_PI_HEIGHT)
        z += percents['beatz'] * shift
    if percents.get('digitalz', 0.0) != 0.0:
        z += _digital_kernel(percents, y_offset, 'digitalz', False)
    if percents.get('tandigitalz', 0.0) != 0.0:
        z += _digital_kernel(percents, y_offset, 'tandigitalz', True)
    if percents.get('squarez', 0.0) != 0.0:
        z += _square_kernel(percents, y_offset, 'squarez')
    if percents.get('bouncez', 0.0) != 0.0:
        z += _bounce_kernel(percents, y_offset, 'bouncez')
    return z


# --- rotations (GetRotationX/Y/Z, :1036) ----------------------------------

def _confusion_axis_degrees(percents, col, beat_now, base):
    """ReceptorGetRotationX/Y/Z (:1091/1115/1140): per-column m_fConfusion*
    constant + <base>offset constant + (songBeat*<base> wrapped)*-180/PI."""
    rot = 0.0
    per_col = percents.get(f'{base}{col}', 0.0)
    if per_col != 0.0:
        rot += per_col * 180.0 / PI
    off = percents.get(base + 'offset', 0.0)
    if off != 0.0:
        rot += off * 180.0 / PI
    mag = percents.get(base, 0.0)
    if mag != 0.0:
        conf = beat_now * mag
        conf = math.fmod(conf, 2.0 * PI)
        conf *= -180.0 / PI
        rot += conf
    return rot


def get_rotation_x(percents, col, y_offset, is_hold_cap=False):
    """GetRotationX (:1036): confusion_x (about horizontal axis) + roll."""
    rot = 0.0
    if (percents.get('confusionx', 0.0) != 0.0
            or percents.get('confusionxoffset', 0.0) != 0.0
            or percents.get(f'confusionx{col}', 0.0) != 0.0):
        rot += _confusion_axis_degrees(percents, col, _beat(percents), 'confusionx')
    if percents.get('roll', 0.0) != 0.0 and not is_hold_cap:
        rot += percents['roll'] * y_offset / 2.0
    return rot


def get_rotation_y(percents, col, y_offset):
    """GetRotationY (:1052): confusion_y (about vertical axis) + twirl."""
    rot = 0.0
    if (percents.get('confusiony', 0.0) != 0.0
            or percents.get('confusionyoffset', 0.0) != 0.0
            or percents.get(f'confusiony{col}', 0.0) != 0.0):
        rot += _confusion_axis_degrees(percents, col, _beat(percents), 'confusiony')
    if percents.get('twirl', 0.0) != 0.0:
        rot += percents['twirl'] * y_offset / 2.0
    return rot


def get_rotation_z(percents, col, note_beat, beat_now, is_hold_head=False):
    """GetRotationZ (:1067): confusion (in-plane spin) + dizzy."""
    rot = 0.0
    if (percents.get('confusion', 0.0) != 0.0
            or percents.get('confusionoffset', 0.0) != 0.0
            or percents.get(f'confusion{col}', 0.0) != 0.0):
        rot += _confusion_axis_degrees(percents, col, beat_now, 'confusion')
    dizzy = percents.get('dizzy', 0.0)
    dizzy_holds = percents.get('_dizzy_holds', False)
    if dizzy != 0.0 and (dizzy_holds or not is_hold_head):
        d = note_beat - beat_now
        d *= dizzy
        d = math.fmod(d, 2.0 * PI)
        d *= 180.0 / PI
        rot += d
    return rot


# --- visibility (ArrowGetPercentVisible, :1255 + GetAlpha, :1312) ----------

def _center_line(percents):
    """GetCenterLine (:1195): CENTER_LINE_Y / (1 - mini*0.5)."""
    zoom = 1.0 - percents.get('mini', 0.0) * 0.5
    return CENTER_LINE_Y / zoom


def _hidden_sudden(percents):
    return percents.get('hidden', 0.0) * percents.get('sudden', 0.0)


def get_percent_visible(percents, col, keycount, y_offset, t_now, beat_now,
                        y_reverse_offset):
    """ArrowGetPercentVisible (:1255). fYPosWithoutReverse = GetYPos with
    WithReverse=false (tipsy folded in). Stealth uses raw y_offset for its
    y<0 test; non-stealth uses the tipsy'd y_pos. RANDOMVANISH deferred."""
    y_pos_without_reverse = get_y_pos(
        percents, col, keycount, y_offset, t_now, beat_now, y_reverse_offset,
        with_reverse=False)
    stealth_type = percents.get('stealth', 0.0) != 0.0 \
        or percents.get(f'stealth{col}', 0.0) != 0.0
    y_pos = y_offset if stealth_type else y_pos_without_reverse

    if y_pos < 0.0:
        return 1.0

    hidden = percents.get('hidden', 0.0)
    sudden = percents.get('sudden', 0.0)
    stealth = percents.get('stealth', 0.0)
    hidden_off = percents.get('hiddenoffset', 0.0)
    sudden_off = percents.get('suddenoffset', 0.0)
    blink = percents.get('blink', 0.0)
    per_col_stealth = percents.get(f'stealth{col}', 0.0)

    center = _center_line(percents)
    hs = _hidden_sudden(percents)
    hidden_end = center + FADE_DIST_Y * _scale(hs, 0.0, 1.0, -1.0, -1.25) \
        + center * hidden_off
    hidden_start = center + FADE_DIST_Y * _scale(hs, 0.0, 1.0, 0.0, -0.25) \
        + center * hidden_off
    sudden_end = center + FADE_DIST_Y * _scale(hs, 0.0, 1.0, -0.0, 0.25) \
        + center * sudden_off
    sudden_start = center + FADE_DIST_Y * _scale(hs, 0.0, 1.0, 1.0, 1.25) \
        + center * sudden_off

    adjust = 0.0
    if hidden != 0.0:
        ha = _clamp(_scale(y_pos, hidden_start, hidden_end, 0.0, -1.0), -1.0, 0.0)
        adjust += hidden * ha
    if sudden != 0.0:
        sa = _clamp(_scale(y_pos, sudden_start, sudden_end, -1.0, 0.0), -1.0, 0.0)
        adjust += sudden * sa
    if stealth != 0.0:
        adjust -= stealth
    if per_col_stealth != 0.0:
        adjust -= per_col_stealth
    if blink != 0.0:
        f = math.sin(t_now * 10.0)
        f = _quantize(f, BLINK_MOD_FREQUENCY)
        adjust += _scale(f, 0.0, 1.0, -1.0, 0.0)
    return _clamp(1.0 + adjust, 0.0, 1.0)


def get_alpha(percents, col, keycount, y_offset, t_now, beat_now,
              y_reverse_offset):
    """GetAlpha (:1312): the draw-distance fade is a separate ramp; below
    it, the hard 0.5 cutoff. Returns 1 or 0 (the engine's hard cut). The
    draw-distance fade is the renderer's culling job here, not modeled."""
    visible = get_percent_visible(percents, col, keycount, y_offset, t_now,
                                  beat_now, y_reverse_offset)
    return 1.0 if visible > 0.5 else 0.0


def get_percent_visible_raw(percents, col, keycount, y_offset, t_now, beat_now,
                            y_reverse_offset):
    """The pre-cutoff visibility in [0,1]. The production port returns this
    smooth value (it composites in float rather than the engine's 0/1),
    documented on `alpha_from_visible`. The harness compares this against
    the production `alpha_mult`; `get_alpha` is the true engine hard cut."""
    return get_percent_visible(percents, col, keycount, y_offset, t_now,
                               beat_now, y_reverse_offset)


# --- zoom (GetZoom, :1571 + GetZoomVariable, :1594) -----------------------

def get_zoom(percents, col, y_offset):
    """ArrowEffects::GetZoom (:1571): NotefieldZoom(1) -> pulse/shrink ->
    tiny (0.5^tiny) + per-column tiny. NOTE: mini is NOT in GetZoom - mini
    is a notefield-level zoom applied by ScreenGameplay, affecting spacing
    and the center line, NOT the per-note sprite zoom. The production port
    folds a per-note mini zoom (1 - mini*0.5) into `zoom`, an APPROXIMATION
    flagged in the harness (item 80)."""
    zoom = 1.0
    zoom = get_zoom_variable(percents, col, y_offset, zoom)
    tiny = percents.get('tiny', 0.0)
    if tiny != 0.0:
        zoom *= math.pow(0.5, tiny)
    per_col_tiny = percents.get(f'tiny{col}', 0.0)
    if per_col_tiny != 0.0:
        zoom *= math.pow(0.5, per_col_tiny)
    return zoom


def get_zoom_variable(percents, col, y_offset, cur_zoom):
    """GetZoomVariable (:1594): pulse then shrink_mult then shrink_linear."""
    zoom = cur_zoom
    pulse_inner = percents.get('pulseinner', 0.0)
    pulse_outer = percents.get('pulseouter', 0.0)
    if pulse_inner != 0.0 or pulse_outer != 0.0:
        offset = percents.get('pulseoffset', 0.0)
        period = percents.get('pulseperiod', 0.0)
        sine = math.sin((y_offset + 100.0 * offset)
                        / (0.4 * (ARROW_SIZE + period * ARROW_SIZE)))
        inner = (pulse_inner * 0.5) + 1.0
        if inner == 0.0:
            inner = 0.01
        zoom *= (sine * (pulse_outer * 0.5)) + inner
    shrink_mult = percents.get('shrinkmult', 0.0)
    if shrink_mult != 0.0 and y_offset >= 0.0:
        zoom *= 1.0 / (1.0 + y_offset * (shrink_mult / 100.0))
    shrink_linear = percents.get('shrinklinear', 0.0)
    if shrink_linear != 0.0 and y_offset >= 0.0:
        zoom += y_offset * (0.5 * shrink_linear / ARROW_SIZE)
    return zoom


# --- time / beat helpers (the harness feeds song time to both sides) ------

def _time(percents):
    return percents.get('_t_now', 0.0)


def _beat(percents):
    return percents.get('_beat_now', 0.0)
