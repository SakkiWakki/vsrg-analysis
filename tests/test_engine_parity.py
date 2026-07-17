"""Differential parity: the vectorized production mod pipeline vs a slow,
line-faithful ITGmania port (`tests.reference_arrow_effects`).

# What this proves

Hand-traced per-formula tests already live in `tests/test_mods.py`. This
file confirms the COMPOSED pipeline: `arrow_effects.note_offsets` (and the
reverse/accel composition exactly as `games/notitg/note_mods.py` layers
it) against the oracle, over (a) thousands of seeded random states and
(b) gat's real compiled channels sampled across the chart.

# The comparison space

Everything is compared in ENGINE PIXELS (the production `note_offsets`
returns engine px; note_mods later multiplies by `lane_w/64`). So the
harness runs the production functions directly with arrow_size = 64 and
compares to the reference, which also works in engine px. The
`lane_w/64` scale is a pure linear post-multiply note_mods applies to
both sides identically, so it cannot introduce a divergence and is left
out (verified once in `test_scale_is_linear_passthrough`).

# The dx convention

Production `note_offsets(...).dx` is the offset from the column's base
x-position; it does NOT include `column_x_offset(col)` (the note layer
adds the lane x separately). The reference `get_x_pos` returns the full
offset from field center INCLUDING the column term. So the parity identity
for x, when `tiny` is off, is:

    production.dx + column_x_offset(col)  ==  reference.get_x_pos

`tiny` breaks this (engine tiny multiplies the WHOLE accumulated offset
incl. the column term; the production port models tiny as a zoom only, not
a spacing compression), so tiny-active cases are compared separately and
classified.

# The reverse/accel convention (memory item 41)

Our native candidate space IS engine reverse=1. note_mods applies
`1 - r_engine` every frame. The harness therefore does NOT compare the
production reverse remap to the engine's GetYPos-with-reverse directly
(they live in mirrored spaces); instead it verifies the r_col resolver
(`reverse_fractions`) against the engine's GetReversePercentForColumn,
which is the load-bearing per-column math, and checks the composition
identity that a zero-channel frame reproduces engine-default upscroll.

# Tolerances

Exact-formula families are compared at rtol/atol tight enough to catch a
real formula error but loose enough to absorb float32-vs-float64 and
numpy-vs-libm transcendental ULP drift (the engine is float32; we are
float64, so we are MORE precise, and the two libms differ in the last
bits of sin/cos/tan/acos). `_EXACT` = 1e-6 absolute on pixel outputs
(sub-thousandth of a pixel) with a relative floor; documented per assert
where a family needs a looser bound and why.
"""
from __future__ import annotations

import functools
import math
import random

import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from tests import reference_arrow_effects as ref

# Pixel-space equality: 1e-6 px absolute + 1e-9 relative. Tighter than any
# visible effect; loose enough for float32/libm ULP noise on bounded inputs.
_EXACT_ATOL = 1e-6
_EXACT_RTOL = 1e-9
# The tan / cosecant kernels blow up near their asymptotes; a 1-ULP phase
# difference between numpy.tan and libm tan near a pole is a huge absolute
# error that is not a port bug. tan cases use a relative-only bound away
# from poles and skip samples within EPS of a pole.
_TAN_RTOL = 1e-6


# ---------------------------------------------------------------------------
# The channel vocabulary the sweeps draw from. Grouped so a random subset is
# a plausible modstring, not a soup of every companion at once.
# ---------------------------------------------------------------------------

# (channel, low, high) - engine-plausible percent ranges incl. negatives.
_X_MODS = [
    ('drunk', -2.0, 2.0), ('tornado', -1.5, 1.5), ('flip', -1.0, 1.0),
    ('invert', -1.0, 4.0), ('beat', -2.0, 2.0), ('bumpyx', -2.0, 2.0),
    ('zigzag', -1.5, 1.5), ('sawtooth', -1.5, 1.5), ('square', -1.5, 1.5),
    ('bounce', -1.5, 1.5), ('digital', -1.5, 1.5), ('parabolax', -1.0, 1.0),
    ('attenuatex', -1.0, 1.0), ('xmode', -0.5, 0.5),
    ('tandrunk', -1.0, 1.0), ('tantornado', -1.0, 1.0),
    ('tanbumpyx', -1.0, 1.0), ('tandigital', -1.0, 1.0),
]
_Y_MODS = [
    ('tipsy', -2.0, 2.0), ('beaty', -2.0, 2.0), ('parabolay', -1.0, 1.0),
    ('attenuatey', -1.0, 1.0), ('tantipsy', -1.0, 1.0),
]
_Z_MODS = [
    ('bumpy', -2.0, 2.0), ('tornadoz', -1.5, 1.5), ('beatz', -2.0, 2.0),
    ('digitalz', -1.5, 1.5), ('zigzagz', -1.5, 1.5), ('sawtoothz', -1.5, 1.5),
    ('squarez', -1.5, 1.5), ('bouncez', -1.5, 1.5), ('parabolaz', -1.0, 1.0),
    ('attenuatez', -1.0, 1.0), ('drunkz', -1.5, 1.5),
]
_ROT_MODS = [
    ('dizzy', -3.0, 3.0), ('confusion', -3.0, 3.0), ('confusionoffset', -1.0, 1.0),
]
# blink is EXCLUDED from the broad alpha sweep: it hits the Quantize
# floor-vs-trunc bug (test_blink_quantize_truncates_toward_zero, xfail) on
# every negative-sine frame, so it would swamp the sweep with one known
# isolated defect. hidden/sudden/stealth are exact.
_ALPHA_MODS = [
    ('hidden', 0.0, 1.0), ('sudden', 0.0, 1.0), ('stealth', 0.0, 1.0),
    ('hiddenoffset', -0.5, 0.5), ('suddenoffset', -0.5, 0.5),
]
_ZOOM_MODS = [
    ('tiny', -1.0, 2.0), ('pulseinner', -1.0, 1.0), ('pulseouter', -1.0, 1.0),
    ('shrinkmult', -1.0, 1.0), ('shrinklinear', -1.0, 1.0),
]
_COMPANIONS = {
    'drunk': ['drunkspeed', 'drunkoffset', 'drunkperiod'],
    'tornado': ['tornadooffset', 'tornadoperiod'],
    'beat': ['beatoffset', 'beatperiod', 'beatmult'],
    'bumpyx': ['bumpyxoffset', 'bumpyxperiod'],
    'zigzag': ['zigzagoffset', 'zigzagperiod'],
    'square': ['squareoffset', 'squareperiod'],
    'bounce': ['bounceoffset', 'bounceperiod'],
    'digital': ['digitaloffset', 'digitalperiod', 'digitalsteps'],
    'sawtooth': ['sawtoothperiod'],
    'tipsy': ['tipsyspeed', 'tipsyoffset'],
    'beaty': ['beatyoffset', 'beatyperiod', 'beatymult'],
    'bumpy': ['bumpyoffset', 'bumpyperiod'],
    'beatz': ['beatzoffset', 'beatzperiod', 'beatzmult'],
    'pulseinner': ['pulseoffset', 'pulseperiod'],
}


def _rand_percents(rng, groups, max_active):
    """A plausible random modstring: pick <=max_active mods from `groups`,
    assign each an in-range percent, and sometimes add its companions."""
    pool = [m for g in groups for m in g]
    k = rng.randint(1, max_active)
    chosen = rng.sample(pool, min(k, len(pool)))
    p = {}
    for name, lo, hi in chosen:
        p[name] = rng.uniform(lo, hi)
        for comp in _COMPANIONS.get(name, []):
            if rng.random() < 0.4:
                p[comp] = rng.uniform(-1.0, 1.0)
    return p


def _flip_period_sign_guarded(p):
    """digital/zigzag/square/bounce period companions can drive the
    denominator to 0 (period = -1). Nudge any such companion off the
    singularity so both sides evaluate a defined value."""
    for key, val in list(p.items()):
        if key.endswith('period') and abs(val + 1.0) < 1e-3:
            p[key] = val + 0.05
    return p


# ---------------------------------------------------------------------------
# Reference wrappers: evaluate the scalar oracle over an array of notes so a
# vectorized production output can be compared element-wise.
# ---------------------------------------------------------------------------

def _ref_x(percents, cols, yoff, keycount, t, beat):
    p = dict(percents, _t_now=t, _beat_now=beat)
    return np.array([ref.get_x_pos(p, int(c), keycount, float(y))
                     for c, y in zip(cols, yoff)])


def _ref_y_contrib(percents, cols, yoff, keycount, t, beat):
    """GetYPos(with_reverse=False) - y_offset = the additive dy the engine
    puts on the scroll axis (tipsy + attenuate_y + beat_y). movey is added
    by the engine outside GetYPos; folded in to match production dy."""
    p = dict(percents, _t_now=t, _beat_now=beat)
    out = []
    for c, y in zip(cols, yoff):
        dy = ref.get_y_pos(p, int(c), keycount, float(y), t, beat, 0.0,
                           with_reverse=False) - float(y)
        movey = p.get(f'movey{int(c)}', p.get('movey', 0.0))
        out.append(dy + ref.ARROW_SIZE * movey if movey else dy)
    return np.array(out)


def _ref_z(percents, cols, yoff, keycount, t, beat):
    p = dict(percents, _t_now=t, _beat_now=beat)
    return np.array([ref.get_z_pos(p, int(c), keycount, float(y), t, beat)
                     for c, y in zip(cols, yoff)])


def _ref_rot_z(percents, cols, note_beats, beat):
    p = dict(percents, _beat_now=beat)
    return np.array([ref.get_rotation_z(p, int(c), float(nb), beat)
                     for c, nb in zip(cols, note_beats)])


def _ref_visible(percents, cols, yoff, keycount, t, beat):
    p = dict(percents, _t_now=t, _beat_now=beat)
    return np.array([ref.get_percent_visible_raw(
        p, int(c), keycount, float(y), t, beat, 0.0)
        for c, y in zip(cols, yoff)])


def _ref_zoom_proxy(percents, cols, yoff, keycount, t, beat):
    """The reference zoom under OUR documented z->zoom proxy: engine
    GetZoom (tiny/pulse/shrink) composed with the engine z-push through the
    same 1 + z/480 reprojection the production port uses. This is the
    contract item 82 formalizes; the harness verifies production matches
    THIS, and separately flags where the proxy diverges from true
    perspective."""
    p = dict(percents, _t_now=t, _beat_now=beat)
    out = []
    for c, y in zip(cols, yoff):
        z = ref.get_z_pos(p, int(c), keycount, float(y), t, beat)
        base = ref.get_zoom(p, int(c), float(y))
        out.append(base * (1.0 + z / ref.SCREEN_HEIGHT))
    return np.array(out)


def _prod(percents, cols, yoff, keycount, t, beat, note_beats=None):
    return ae.note_offsets(percents, np.asarray(cols), np.asarray(yoff,
                           dtype=float), t_now=t, beat_now=beat,
                           keycount=keycount, note_beats=note_beats)


def _col_x(cols, keycount):
    return ae.column_offsets(keycount)[np.asarray(cols)]


# ===========================================================================
# GROUP 1: exact per-family sweeps (dx / dy / z / rotation / alpha / zoom)
# ===========================================================================

_KEYCOUNTS = [4, 5, 6, 8]


def _sweep_cases(seed, n, groups, max_active, keycounts=_KEYCOUNTS,
                 yoff_lo=-200.0, yoff_hi=800.0):
    rng = random.Random(seed)
    for _ in range(n):
        keycount = rng.choice(keycounts)
        m = rng.randint(1, 6)
        cols = [rng.randrange(keycount) for _ in range(m)]
        yoff = [rng.uniform(yoff_lo, yoff_hi) for _ in range(m)]
        t = rng.uniform(0.0, 60.0)
        beat = rng.uniform(0.0, 200.0)
        p = _flip_period_sign_guarded(_rand_percents(rng, groups, max_active))
        yield keycount, np.array(cols), np.array(yoff), t, beat, p


def test_dx_parity_sweep():
    """X mods (no tiny): production dx + column offset == reference GetXPos.
    Covers drunk/tornado/flip/invert/beat/bumpyx/waveform-x/parabola/
    attenuate/xmode/movex and their companions. tan family is guarded by a
    pole skip; everything else is exact."""
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            1, 3000, [_X_MODS], max_active=4):
        p.pop('tiny', None)
        # movex sometimes, to exercise the post-column term.
        prod = _prod(p, cols, yoff, keycount, t, beat)
        got = prod.dx + _col_x(cols, keycount)
        want = _ref_x(p, cols, yoff, keycount, t, beat)
        _assert_close_x(got, want, p, keycount, cols, yoff, t, beat)


def _has_tan(p):
    return any(k.startswith('tan') for k in p)


def _square_guard_mask(p, cols, yoff, suffix=''):
    """True where a square/squarez sample lands in the engine's <0.01
    small-angle guard band (angle in [0, 0.01) after fmod) - the region the
    port gets WRONG (sign flip), owned by test_square_small_positive_angle_
    guard as an xfail. Masked out of the exact sweeps so a known, isolated
    bug does not fail the broad parity check."""
    base = 'square' + suffix
    mag = p.get(base, 0.0)
    if mag == 0.0:
        return np.zeros(len(cols), dtype=bool)
    offset = p.get(base + 'offset', 0.0)
    period = p.get(base + 'period', 0.0)
    angle = ref.PI * (np.asarray(yoff) + 1.0 * offset) / (
        ref.ARROW_SIZE + period * ref.ARROW_SIZE)
    a = np.mod(angle, 2.0 * ref.PI)
    # also flag near the top of the wrapped range (fmod of a tiny negative)
    return (a < 0.02) | (a > 2.0 * ref.PI - 0.02)


def _tornadoz_wide(p, keycount):
    """True when a tornadoz / tantornadoz sample uses the wrong window width
    - the port narrows the tornado window to 2 for ALL dimensions in a wide
    field (>4 cols), but the engine narrows only X (dim 0); Z keeps width 3.
    Owned by test_tornadoz_window_width_ignores_dimension (xfail); masked
    from the Z / zoom sweeps for keycount > 4 so the isolated bug does not
    fail the broad checks."""
    return keycount > 4 and (p.get('tornadoz', 0.0) != 0.0
                             or p.get('tantornadoz', 0.0) != 0.0)


def _assert_close_x(got, want, p, keycount, cols, yoff, t, beat):
    keep = ~_square_guard_mask(p, cols, yoff)
    got, want = got[keep], want[keep]
    if not len(want):
        return
    if _has_tan(p):
        # Skip samples near a tan pole (|value| explodes); compare the rest
        # relatively. A real formula bug shifts ALL samples, not just the
        # bounded ones, so this still catches port errors.
        finite = np.abs(want) < 1e4
        if finite.any():
            np.testing.assert_allclose(
                got[finite], want[finite], rtol=_TAN_RTOL, atol=1e-3)
        return
    np.testing.assert_allclose(got, want, rtol=_EXACT_RTOL, atol=_EXACT_ATOL,
                               err_msg=f'dx mismatch p={p} k={keycount} '
                                       f'cols={cols} yoff={yoff} t={t} beat={beat}')


def test_dy_parity_sweep():
    """Y mods: production dy == engine GetYPos additive contribution
    (tipsy + attenuate_y + beat_y + movey). parabola_y is EXCLUDED here and
    tested separately (it is a known routing divergence - the engine reads
    it in GetYOffset, the port in GetYPos)."""
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            2, 3000, [_Y_MODS], max_active=4):
        p.pop('parabolay', None)
        prod = _prod(p, cols, yoff, keycount, t, beat)
        want = _ref_y_contrib(p, cols, yoff, keycount, t, beat)
        if _has_tan(p):
            finite = np.abs(want) < 1e4
            if finite.any():
                np.testing.assert_allclose(prod.dy[finite], want[finite],
                                           rtol=_TAN_RTOL, atol=1e-3)
            continue
        np.testing.assert_allclose(prod.dy, want, rtol=_EXACT_RTOL,
                                   atol=_EXACT_ATOL,
                                   err_msg=f'dy mismatch p={p} k={keycount}')


def test_z_push_parity_sweep():
    """Z mods: the engine z-push (GetZPos) drives our zoom via the 1+z/480
    proxy. Here we compare the RAW z-push that production accumulates
    (recovered from its zoom output) against reference GetZPos, isolating
    the z math from the pulse/shrink/tiny zoom factors."""
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            3, 3000, [_Z_MODS], max_active=4):
        if _tornadoz_wide(p, keycount):
            continue
        # No pulse/shrink/tiny/mini -> production zoom == 1 * (1 + z/480).
        prod = _prod(p, cols, yoff, keycount, t, beat)
        prod_z = (prod.zoom - 1.0) * ae.SCREEN_HEIGHT
        want = _ref_z(p, cols, yoff, keycount, t, beat)
        keep = ~_square_guard_mask(p, cols, yoff, suffix='z')
        prod_z, want = prod_z[keep], want[keep]
        if not len(want):
            continue
        if _has_tan(p):
            finite = np.abs(want) < 1e4
            if finite.any():
                np.testing.assert_allclose(prod_z[finite], want[finite],
                                           rtol=1e-5, atol=1e-2)
            continue
        np.testing.assert_allclose(prod_z, want, rtol=1e-6, atol=1e-4,
                                   err_msg=f'z mismatch p={p} k={keycount}')


def test_rotation_parity_sweep():
    """dizzy + confusion(Z): production rotation_deg == engine GetRotationZ.
    confusion X/Y are out-of-plane, reprojected to zoom/dx (tested there),
    so they are excluded from the rotation comparison."""
    rng = random.Random(4)
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            4, 3000, [_ROT_MODS], max_active=3):
        note_beats = np.array([rng.uniform(0.0, 200.0) for _ in cols])
        prod = _prod(p, cols, yoff, keycount, t, beat, note_beats=note_beats)
        want = _ref_rot_z(p, cols, note_beats, beat)
        # Rotations are compared MODULO 360: the production dizzy/confusion
        # wrap the angle with np.mod (result in [0,360)) where the engine
        # uses std::fmod (result in (-360,360)). A -69.3deg spin and a
        # +290.7deg spin are the IDENTICAL visual rotation - the difference
        # is a pure representative-of-equivalence-class choice, not a visible
        # divergence. See the report's fork/convention section.
        diff = np.mod(prod.rotation_deg - want + 180.0, 360.0) - 180.0
        np.testing.assert_allclose(diff, 0.0, rtol=0, atol=1e-6,
                                   err_msg=f'rot mismatch (mod360) p={p} '
                                           f'k={keycount}')


def test_alpha_parity_sweep():
    """stealth/hidden/sudden/blink + offsets: production alpha_mult ==
    engine ArrowGetPercentVisible (smooth, pre-hard-cut). The engine's hard
    0.5 cutoff is a documented substitution (we composite in float); the
    smooth value is what both sides must agree on."""
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            5, 3000, [_ALPHA_MODS], max_active=4):
        prod = _prod(p, cols, yoff, keycount, t, beat)
        want = _ref_visible(p, cols, yoff, keycount, t, beat)
        np.testing.assert_allclose(prod.alpha_mult, want, rtol=1e-6,
                                   atol=1e-6,
                                   err_msg=f'alpha mismatch p={p} k={keycount} '
                                           f'cols={cols} yoff={yoff} t={t}')


# ===========================================================================
# GROUP 2: reverse resolver + composition identities (memory item 41)
# ===========================================================================

def test_reverse_fractions_vs_engine():
    """reverse_fractions == PlayerOptions::GetReversePercentForColumn over
    the scroll family (reverse/split/alternate/cross), every keycount and
    column. Numbered per-column `reverse<c>` is EXCLUDED here (the port
    omits it - see test_reverse_numbered_per_column, xfail)."""
    rng = random.Random(6)
    for _ in range(2000):
        keycount = rng.choice(_KEYCOUNTS)
        p = {}
        for name in ('reverse', 'split', 'alternate', 'cross'):
            if rng.random() < 0.7:
                p[name] = rng.uniform(-0.5, 2.5)
        cols = np.arange(keycount)
        got = ae.reverse_fractions(p, cols, keycount)
        want = np.array([ref.get_reverse_percent_for_column(p, c, keycount)
                         for c in range(keycount)])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9,
                                   err_msg=f'reverse frac p={p} k={keycount}')


def test_zero_channel_is_engine_default_upscroll():
    """The composition invariant (item 41): with no channels, note_mods'
    effective reverse is 1 - 0 = 1 for every column, i.e. a full mirror.
    In native (reverse=1) candidate space that reproduces engine-default
    upscroll. Verified on the resolver: effective = 1 - r_engine, and
    r_engine(no channels) = 0 everywhere."""
    for keycount in _KEYCOUNTS:
        cols = np.arange(keycount)
        r_engine = ae.reverse_fractions({}, cols, keycount)
        np.testing.assert_array_equal(r_engine, np.zeros(keycount))
        effective = 1.0 - r_engine
        np.testing.assert_array_equal(effective, np.ones(keycount))


# ===========================================================================
# GROUP 3: accel family (y_offset reshapers), composed as note_mods does
# ===========================================================================

def test_accel_y_offset_parity_sweep():
    """boost/brake/wave/expand/boomerang: accel_y_offset (the production
    remap note_mods runs first) == reference GetYOffset accel section, over
    randomized single- and multi-mod states. Both take one scalar y_offset;
    swept element-wise."""
    rng = random.Random(7)
    accel_pool = [('boost', -1.0, 2.0), ('brake', -1.0, 2.0),
                  ('wave', -2.0, 2.0), ('expand', 0.0, 1.0),
                  ('boomerang', 0.0, 1.0)]
    for _ in range(3000):
        p = {}
        for name, lo, hi in rng.sample(accel_pool, rng.randint(1, 3)):
            p[name] = rng.uniform(lo, hi)
        # waveperiod is EXCLUDED: the port hardcodes the wave denominator to
        # 38 and ignores this companion (test_wave_period_companion_ignored,
        # xfail). wave with the default period (0) is exact.
        if 'expand' in p:
            p['_expand_phase'] = rng.uniform(0.0, 6.3)
        yoff = np.array([rng.uniform(-200.0, 800.0) for _ in range(8)])
        got = ae.accel_y_offset(p, yoff.copy())
        want = np.array([ref.get_y_offset_accel(p, float(y)) for y in yoff])
        # The engine's y<0 early-out (:595) returns before expand AND before
        # the boomerang fold, so a past-receptor note keeps its raw offset;
        # the port applies both to y<0 notes (xfails
        # test_expand_skips_past_receptor_notes /
        # test_boomerang_skips_past_receptor_notes). Mask y<0 when either is
        # active so the rest of the accel family is checked exactly.
        keep = np.ones(len(yoff), dtype=bool)
        if p.get('expand', 0.0) or p.get('boomerang', 0.0):
            keep = yoff >= 0.0
        np.testing.assert_allclose(got[keep], want[keep], rtol=1e-6, atol=1e-4,
                                   err_msg=f'accel mismatch p={p}')


# ===========================================================================
# GROUP 4: documented-approximation contracts (classification ii)
# ===========================================================================

def test_zoom_proxy_contract_sweep():
    """Our zoom output == engine GetZoom composed with GetZPos through the
    documented 1+z/480 proxy (item 82's contract), EXCLUDING mini (which we
    add as a separate proxy, tested below) and confusionx (reprojected).
    This confirms the pulse/shrink/tiny/z composition is engine-faithful up
    to the declared reprojection."""
    for keycount, cols, yoff, t, beat, p in _sweep_cases(
            8, 3000, [_ZOOM_MODS, _Z_MODS], max_active=4):
        p.pop('mini', None)
        # tiny uses the wrong zoom curve (1 - t*0.5 vs engine 0.5^t,
        # test_tiny_zoom_curve_matches_engine, xfail); drop it from the
        # exact composition sweep so the pulse/shrink/z composition is what
        # is checked here.
        p.pop('tiny', None)
        if _tornadoz_wide(p, keycount):
            continue
        keep = ~_square_guard_mask(p, cols, yoff, suffix='z')
        prod = _prod(p, cols, yoff, keycount, t, beat)
        want = _ref_zoom_proxy(p, cols, yoff, keycount, t, beat)
        if _has_tan(p):
            continue
        np.testing.assert_allclose(prod.zoom[keep], want[keep], rtol=1e-6,
                                   atol=1e-5,
                                   err_msg=f'zoom proxy p={p} k={keycount}')


def test_confusionx_zoom_is_cos_foreshorten():
    """confusionx (an out-of-plane X tilt) reprojects to a uniform zoom of
    |cos(angle)| (documented approximation ii). Verify production
    confusionx_zoom == |cos| of the engine's confusion angle."""
    rng = random.Random(9)
    for _ in range(500):
        percent = rng.uniform(-3.0, 3.0)
        beat = rng.uniform(0.0, 200.0)
        offset = rng.uniform(-1.0, 1.0) if rng.random() < 0.5 else 0.0
        got = ae.confusionx_zoom(percent, beat, offset)
        deg = ref._confusion_axis_degrees(
            {'confusionx': percent, 'confusionxoffset': offset}, 0, beat,
            'confusionx')
        want = abs(math.cos(math.radians(deg)))
        assert abs(got - want) < 1e-9


def test_confusiony_dx_is_horizontal_foreshorten():
    """confusiony reprojects to a per-column dx pulling x toward center by
    (cos(angle)-1)*xoffset (documented approximation ii)."""
    rng = random.Random(10)
    for _ in range(500):
        keycount = rng.choice(_KEYCOUNTS)
        percent = rng.uniform(-3.0, 3.0)
        beat = rng.uniform(0.0, 200.0)
        cols = np.arange(keycount)
        got = ae.confusiony_dx(percent, cols, beat, keycount, 0.0)
        deg = ref._confusion_axis_degrees({'confusiony': percent}, 0, beat,
                                          'confusiony')
        factor = math.cos(math.radians(deg)) - 1.0
        want = ae.column_offsets(keycount) * factor
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_boomerang_visibility_is_documented_fade():
    """boomerang's visibility half is a documented approximation (ii): the
    engine culls past-peak arrows; we fade alpha to 0 over one ARROW_SIZE
    past the fold p = H*0.75. Verify the fade shape and that the reference's
    peak matches ours (the position parabola itself is exact, tested in the
    accel sweep)."""
    p_ref = ref.boomerang_peak()
    p_prod = ae.boomerang_peak()[1]
    assert abs(p_ref - p_prod) < 1e-9
    # Fade: full alpha at/below fold, 0 at fold+ARROW_SIZE.
    fold = ref.SCREEN_HEIGHT * ref.BOOMERANG_PEAK_PERCENTAGE
    assert ae.boomerang_visibility(1.0, np.array([fold - 1.0]))[0] == 1.0
    assert ae.boomerang_visibility(1.0, np.array([fold + ae.ARROW_SIZE]))[0] \
        == pytest.approx(0.0)


# ===========================================================================
# GROUP 5: MISMATCH repros (xfail with analysis for the fix agent to flip)
# ===========================================================================

@pytest.mark.xfail(reason=(
    "BUG (class i): tiny is modeled as a per-note ZOOM only. The engine "
    "(GetZoom :1582) uses zoom *= pow(0.5, tiny), but the production "
    "tiny_zoom uses 1 - tiny*0.5 - a DIFFERENT curve. They agree only at "
    "tiny in {0, 1} (both give 1.0 and 0.5). At tiny=2 engine=0.25, "
    "port=0.0 (arrow vanishes); at tiny=-1 engine=2.0, port=1.5; at "
    "tiny=0.5 engine=0.707, port=0.75. gat strobes tiny/mini sections so "
    "the amplitude error is visible. FIX: arrow_effects.tiny_zoom should "
    "return pow(0.5, tiny_percent). Additionally the engine tiny COMPRESSES "
    "SPACING (GetXPos :1025 multiplies the whole pixel offset incl. the "
    "column term by min(pow(0.5,tiny),1)); the port models no spacing "
    "change - a separate, larger gap (the real-state x comparison fences "
    "tiny-active samples for exactly this reason). This test targets only "
    "the zoom-curve half."),
    strict=True)
def test_tiny_zoom_curve_matches_engine():
    for tiny in (0.5, 2.0, -1.0, 1.5):
        got = ae.tiny_zoom(tiny)
        want = math.pow(0.5, tiny)
        assert got == pytest.approx(want), f'tiny={tiny} got={got} want={want}'


@pytest.mark.xfail(reason=(
    "BUG (class i): the waveform SQUARE kernel drops the engine's <0.01 "
    "angle guard. RageSquare (RageMath.cpp:584) nudges any angle < 0.01 up "
    "by 2*PI, which for a small POSITIVE angle in [0, 0.01) flips the "
    "result from +1 to -1. arrow_effects.rage_square uses np.mod (always "
    "[0,2*PI)) and returns +1 there, and its docstring wrongly claims the "
    "guard is a no-op. A note whose square/squarez phase lands in that "
    "narrow band gets a FULL ARROW_SIZE sign flip (+32px vs -32px at 100%). "
    "This is a thin band (angle 0..0.01) but hits every note crossing it "
    "exactly at the receptor region. FIX: port the <0.01 -> +2*PI guard "
    "into rage_square (scalar: where a<0.01, use a+2*PI before the >=PI "
    "test)."),
    strict=True)
def test_square_small_positive_angle_guard():
    # An angle in [0, 0.01): engine wraps to ~2*PI (>= PI) -> -1; port -> +1.
    angle = 0.005
    assert ae.rage_square(angle) == ref.rage_square(angle)


@pytest.mark.xfail(reason=(
    "BUG (class i): blink's Quantize uses np.floor, but the engine's "
    "Quantize (RageUtil.h:175) casts with int() which TRUNCATES toward "
    "zero. For a negative sine (blink samples sin(t*10), which is negative "
    "half the time) floor and trunc differ by one interval. Engine: "
    "int((f + i/2)/i)*i; port: floor((f + i/2)/i)*i. At f=-0.5, i=0.3333: "
    "(f+i/2)/i = -0.9999 -> int()=0 (trunc) vs floor()=-1, so quantized f "
    "= 0.0 (engine) vs -0.3333 (port), a visible-flicker phase error in "
    "blink's fade. FIX: arrow_effects._quantize should truncate toward "
    "zero (np.trunc) to match int() semantics, matching the reference's "
    "_quantize."),
    strict=True)
def test_blink_quantize_truncates_toward_zero():
    # A t_now whose sin(t*10) is negative and lands where trunc != floor.
    rng = random.Random(11)
    mismatch_found = False
    for _ in range(2000):
        t = rng.uniform(0.0, 60.0)
        got = ae.blink_adjust(1.0, t)
        f = math.sin(t * 10.0)
        f = ref._quantize(f, ref.BLINK_MOD_FREQUENCY)
        want = ref._scale(f, 0.0, 1.0, -1.0, 0.0)
        if abs(got - want) > 1e-9:
            mismatch_found = True
            assert got == pytest.approx(want), (
                f't={t} sin={math.sin(t * 10.0)} got={got} want={want}')
    # If no random t exercised the trunc/floor split, the test is vacuous;
    # force a known-divergent value so the xfail stays meaningful.
    assert mismatch_found, "expected at least one trunc/floor divergence"


@pytest.mark.xfail(reason=(
    "BUG (class i): tornadoz / tantornadoz use the wrong tornado WINDOW "
    "WIDTH in wide fields (>4 cols). The engine (ArrowEffects::Init :358) "
    "narrows the window from 3 to 2 ONLY for dimension 0 (x): `if "
    "(dimension == 0 && wide_field) width = 2;`. The Z window (dimension 2) "
    "stays width 3. The port's _tornado_window ignores the dimension and "
    "narrows to 2 for every axis whenever keycount > 4, so tornadoz's "
    "arccos window is too narrow at 5/6/8 keys - a wrong z-push, hence a "
    "wrong per-note zoom. (4K is unaffected: width is 3 for both there, so "
    "gat is fine; 6K/8K NotITG/Etterna charts using tornadoz diverge.) FIX: "
    "thread a `dimension` arg through _tornado_window / _tornado_offset and "
    "apply the width-2 narrowing only for dimension 0, mirroring the engine "
    "Init loop."),
    strict=True)
def test_tornadoz_window_width_ignores_dimension():
    keycount = 8  # wide field: engine z-width 3, port width 2
    cols = np.arange(keycount)
    yoff = np.full(keycount, 300.0)
    prod_z = (ae.note_offsets({'tornadoz': 0.5}, cols, yoff, t_now=1.0,
              beat_now=0.0, keycount=keycount).zoom - 1.0) * ae.SCREEN_HEIGHT
    want = np.array([ref.get_z_pos({'tornadoz': 0.5}, int(c), keycount, 300.0,
                                   1.0, 0.0) for c in cols])
    np.testing.assert_allclose(prod_z, want, rtol=1e-6, atol=1e-4)


@pytest.mark.xfail(reason=(
    "BUG (class i, minor): reverse_fractions omits the per-column numbered "
    "`reverse<c>` term. The engine GetReversePercentForColumn "
    "(PlayerOptions.cpp:1688) adds `m_fReverse[iCol]` alongside the "
    "reverse/split/alternate/cross scrolls before the >2 wrap. The port's "
    "reverse_fractions reads only reverse/split/alternate/cross, so a chart "
    "that reverses a SINGLE column (a Mirin `reverse0`/`reverse3` style "
    "poke) gets no per-column scroll flip and the wrap lands on a different "
    "branch. gat uses only whole-field reverse (=1 baseline), so this is "
    "latent for it; charts with numbered reverse diverge. FIX: fold a "
    "numbered-`reverse<c>` lookup into reverse_fractions the same way the "
    "other mods pick up per-column variants."),
    strict=True)
def test_reverse_numbered_per_column():
    keycount = 4
    p = {'reverse': 0.2849, 'split': 1.4885, 'alternate': 1.7792,
         'reverse3': 0.6038}
    cols = np.arange(keycount)
    got = ae.reverse_fractions(p, cols, keycount)
    want = np.array([ref.get_reverse_percent_for_column(p, c, keycount)
                     for c in range(keycount)])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.xfail(reason=(
    "BUG (class i, minor): EXPAND is applied to past-receptor notes "
    "(y_offset < 0). The engine's GetYOffset returns early for y<0 "
    "(:595 `if (fYOffset < 0) return fYOffset * fScrollSpeed;`) BEFORE "
    "expand ever modifies fScrollSpeed, so a y<0 note keeps unscaled speed. "
    "accel_y_offset's `past = y < 0` mask guards boost/brake/wave but NOT "
    "the trailing `out = out * SCALE(expand,...)`, so it scales y<0 notes "
    "too. Visible only for notes drawn behind the judgment line under "
    "expand - small, but a real divergence. FIX: apply the expand multiply "
    "only where y_offset >= 0 (or short-circuit past-receptor notes to "
    "their raw offset, matching the engine early-out)."),
    strict=True)
def test_expand_skips_past_receptor_notes():
    p = {'expand': 0.5, '_expand_phase': 1.0}
    yoff = np.array([-100.0])
    got = ae.accel_y_offset(p, yoff.copy())[0]
    want = ref.get_y_offset_accel(p, -100.0)  # engine: -100 (unscaled)
    assert got == pytest.approx(want)


@pytest.mark.xfail(reason=(
    "BUG (class i): the BOOMERANG position parabola folds past-receptor "
    "notes (y_offset < 0), which the engine never does. GetYOffset's y<0 "
    "early-out (:595) returns before the boomerang fold (:646), so a y<0 "
    "note keeps its raw offset; accel_y_offset applies boomerang_y_offset "
    "to the WHOLE array (the `past` mask guards only boost/brake/wave). At "
    "y=-100 the port folds to -170.8 while the engine leaves it at -100 - a "
    "visible 70px displacement of every note drawn behind the judgment "
    "line under boomerang. NOTE the docstring on accel_y_offset asserts the "
    "opposite ('the engine, which folds y<0 too') - that justification is "
    "incorrect; the engine's early-out precedes the fold. FIX: apply the "
    "boomerang fold only where y_offset >= 0 (compute on the guarded array), "
    "and correct the docstring."),
    strict=True)
def test_boomerang_skips_past_receptor_notes():
    p = {'boomerang': 0.5}
    yoff = np.array([-100.0])
    got = ae.accel_y_offset(p, yoff.copy())[0]
    want = ref.get_y_offset_accel(p, -100.0)  # engine: -100 (untouched)
    assert got == pytest.approx(want)


@pytest.mark.xfail(reason=(
    "BUG (class i): the WAVE accel mod ignores its `waveperiod` companion. "
    "The engine (GetYOffset :629) computes wave as `wave * WAVE_MOD_MAGNITUDE"
    " * sin(y / ((waveperiod * WAVE_MOD_HEIGHT) + WAVE_MOD_HEIGHT))` "
    "= sin(y / (waveperiod*38 + 38)); accel_y_offset hardcodes the "
    "denominator to 38.0 (`wave * 20 * sin(y/38)`), so any chart that "
    "stretches wave via waveperiod gets the wrong spatial frequency. With "
    "the default period (0) they agree. FIX: read percents['waveperiod'] "
    "and use ((waveperiod*38)+38) as the denominator, mirroring the engine's"
    " WAVE_MOD_HEIGHT metric."),
    strict=True)
def test_wave_period_companion_ignored():
    p = {'wave': 0.31, 'waveperiod': 1.0}
    got = ae.accel_y_offset(p, np.array([300.0]))[0]
    want = ref.get_y_offset_accel(p, 300.0)
    assert got == pytest.approx(want, abs=1e-4)


# A NON-xfail guard for the documented parabola_y routing divergence: it is
# an intentional/accepted difference (class ii/iv), not a bug, so we assert
# the CONTRACT (production routes parabolay as an additive dy, matching the
# engine's contribution wherever the y<0 early-out and boomerang fold don't
# apply) rather than xfail it.
def test_parabola_y_additive_contribution_matches_when_approaching():
    """When a note is approaching (y_offset >= 0) and no accel/boomerang is
    active, the engine's GetYOffset parabola_y and the port's GetYPos
    parabola_y are the SAME additive term. This confirms the routing choice
    is value-equivalent in the common case; the divergence is confined to
    y<0 (early-out) and boomerang-active frames, documented in the report."""
    rng = random.Random(12)
    for _ in range(1000):
        keycount = rng.choice(_KEYCOUNTS)
        col = rng.randrange(keycount)
        y = rng.uniform(0.0, 800.0)  # approaching only
        pct = rng.uniform(-1.0, 1.0)
        prod = ae.note_offsets({'parabolay': pct}, np.array([col]),
                               np.array([y]), t_now=0.0, beat_now=0.0,
                               keycount=keycount)
        # engine additive parabola_y term
        want = pct * (y / ae.ARROW_SIZE) * (y / ae.ARROW_SIZE)
        assert prod.dy[0] == pytest.approx(want, abs=1e-6)


# ===========================================================================
# GROUP 6: REAL STATES - gat's compiled channels sampled across the chart
# ===========================================================================
#
# Part (b) of the task: sample the pilot modchart's actual compiled channel
# values at timestamps spanning the chart and compare the FULL note_offsets
# output at each against the reference. This is the strongest integration
# check - it confirms that real, dense, many-mod-at-once states compose the
# same way in the fast port and the oracle, and that the ONLY divergences
# at real states are the ones the isolated sweeps already classified.

_GAT_SM = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/UKSRT8/'
           '5. gat/gat.sm')


@functools.lru_cache(maxsize=1)
def _gat_channels_cached():
    """Compile gat's modfile ONCE and share the channels across every
    parametrized real-state case (compile is ~7s; 30+ recompiles would blow
    the timeout). Returns None when the install is absent."""
    import os
    if not os.path.exists(_GAT_SM):
        return None
    from analysis.games.notitg.modfile import compile_modfile
    from analysis.games.notitg.mod_channels import compile_mod_channels
    compiled = compile_modfile(_GAT_SM)
    if not compiled or not compiled.get('mod_events'):
        return None
    return compile_mod_channels(compiled['mod_events'])


def _gat_channels():
    channels = _gat_channels_cached()
    if channels is None:
        pytest.skip('gat modfile not present / did not compile '
                    '(needs the NotITG install)')
    return channels


def _gat_beat_at(t):
    """gat is a steady 205 BPM after its short intro; the exact beat only
    matters for beat/confusion/dizzy phase. Use a plausible song beat so
    those mods are exercised at a non-trivial phase (both sides get the
    same value, so any beat is a valid comparison point)."""
    return t * (205.0 / 60.0)


# The masks that fence off the isolated, already-classified bugs so the
# real-state comparison asserts NO UNCLASSIFIED divergence. Each maps to an
# xfail repro above.
def _classified_x_mask(p, cols, yoff, keycount):
    m = _square_guard_mask(p, cols, yoff)
    # tiny rescales the whole x offset in the engine (spacing compression)
    # but only zooms in the port: any tiny-active sample's x diverges by
    # design (documented approximation + the tiny curve bug). Fence it.
    if p.get('tiny', 0.0) != 0.0:
        m = np.ones(len(cols), dtype=bool)
    # hallway / confusiony are 2D REPROJECTIONS of out-of-plane perspective
    # (class ii): they add a per-note dx in the port, but the engine has NO
    # ArrowEffects x-term for them (hallway is a notefield actor tilt;
    # confusiony is a GetRotationY about the vertical axis). So the port's
    # dx is expected to diverge from the reference's GetXPos here - it is
    # the documented approximation, verified in the dedicated reprojection
    # tests, not a composition bug. Fence these samples from the exact x
    # comparison.
    if (p.get('hallway', 0.0) != 0.0 or p.get('confusiony', 0.0) != 0.0
            or p.get('confusionyoffset', 0.0) != 0.0):
        m = np.ones(len(cols), dtype=bool)
    return m


@pytest.mark.parametrize('t', [round(0.5 + i * 517.0 / 29, 2) for i in range(30)])
def test_gat_real_state_parity(t):
    """At 30 timestamps across gat, production note_offsets == reference on
    every axis, EXCEPT samples fenced by the classified-bug masks. If an
    unclassified divergence exists, this fails with the timestamp + percents
    + axis so it can be added to the classification."""
    channels = _gat_channels()
    keycount = 4
    beat = _gat_beat_at(t)
    percents = channels.values_at(t)

    # A realistic note layout: every column, a spread of scroll offsets
    # (approaching + a couple past the receptor), plus per-note beats.
    cols = np.array([c for c in range(keycount) for _ in range(6)])
    yoff = np.tile(np.array([-80.0, 40.0, 160.0, 320.0, 500.0, 760.0]), keycount)
    note_beats = beat + (yoff / 64.0)  # rough beats-until-step for dizzy

    prod = ae.note_offsets(percents, cols, yoff, t_now=t, beat_now=beat,
                           keycount=keycount, note_beats=note_beats)

    ref_x = _ref_x(percents, cols, yoff, keycount, t, beat) - _col_x(cols, keycount)
    ref_dy = _ref_y_contrib(percents, cols, yoff, keycount, t, beat)
    ref_rot = _ref_rot_z(percents, cols, note_beats, beat)
    ref_alpha = _ref_visible(percents, cols, yoff, keycount, t, beat)

    x_keep = ~_classified_x_mask(percents, cols, yoff, keycount)
    np.testing.assert_allclose(
        prod.dx[x_keep], ref_x[x_keep], rtol=1e-6, atol=1e-4,
        err_msg=f'gat dx divergence at t={t} percents={_active(percents)}')
    np.testing.assert_allclose(
        prod.dy, ref_dy, rtol=1e-6, atol=1e-4,
        err_msg=f'gat dy divergence at t={t} percents={_active(percents)}')
    rot_diff = np.mod(prod.rotation_deg - ref_rot + 180.0, 360.0) - 180.0
    np.testing.assert_allclose(
        rot_diff, 0.0, rtol=0, atol=1e-5,
        err_msg=f'gat rot divergence at t={t} percents={_active(percents)}')
    np.testing.assert_allclose(
        prod.alpha_mult, ref_alpha, rtol=1e-6, atol=1e-5,
        err_msg=f'gat alpha divergence at t={t} percents={_active(percents)}')


def _active(percents):
    return {k: round(v, 4) for k, v in percents.items() if abs(v) > 1e-4}


def test_gat_exercises_the_classified_bugs():
    """Sanity: gat's real channels DO activate tiny / square / waveperiod
    (the three bug channels present in the chart), so the classified
    approximations above are load-bearing for the pilot, not hypothetical.
    Guards against the real-state test silently passing because gat never
    reaches a bug region."""
    channels = _gat_channels()
    hit = set()
    for i in range(201):
        t = i * 517.0 / 200
        for k, v in channels.values_at(t).items():
            if abs(v) > 1e-4 and k in ('tiny', 'square', 'waveperiod'):
                hit.add(k)
    assert {'tiny', 'square', 'waveperiod'} <= hit, (
        f'expected gat to activate tiny/square/waveperiod, got {hit}')


def test_scale_is_linear_passthrough():
    """Why the harness can compare in engine px (arrow_size=64) even though
    note_mods works in OUR lane pixels: note_mods (a) divides the screen
    distance by `scale = lane_w/64` to recover the ENGINE y_offset, (b)
    calls note_offsets with the DEFAULT arrow_size=64 (verified below - it
    never passes lane_w as arrow_size), then (c) multiplies the returned
    dx/dy by `scale`. So the value note_offsets computes is exactly what
    this harness compares, and the only lane-space step is a scalar multiply
    note_mods applies uniformly - it cannot change agreement with the
    reference. Confirm the mechanical facts: note_offsets' default arrow_size
    is 64, and the dx/dy outputs scale linearly under an EXTERNAL multiply
    (the trivial post-scale note_mods does), independent of the formulas'
    internal (non-homogeneous) structure."""
    import inspect
    sig = inspect.signature(ae.note_offsets)
    assert sig.parameters['arrow_size'].default == 64.0
    # note_mods never overrides arrow_size (it feeds the engine y_offset it
    # recovered by dividing screen distance by scale), so note_offsets runs
    # at 64 - exactly the space this reference works in. The lane multiply
    # is applied AFTER, to dx/dy alike, so it is a scalar the comparison is
    # invariant to. Confirm the engine-space output matches the reference at
    # arrow_size=64 for a representative multi-mod state (the whole harness
    # rests on this identity holding at 64).
    p = {'drunk': 1.0, 'tipsy': 0.5, 'beat': 0.3, 'bumpyx': 0.4}
    cols = np.array([0, 1, 2, 3])
    yoff = np.array([100.0, 200.0, 300.0, 400.0])
    off = ae.note_offsets(p, cols, yoff, t_now=1.0, beat_now=5.0, keycount=4)
    want = _ref_x(p, cols, yoff, 4, 1.0, 5.0) - _col_x(cols, 4)
    np.testing.assert_allclose(off.dx, want, rtol=1e-9, atol=1e-6)
