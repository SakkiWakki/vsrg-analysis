"""Curve-based assembler for per-note mod geometry.

Reproduces the POSITION + ROTATION axes of `arrow_effects.note_offsets`
(dx, dy, z, rotation_deg, rot_x, rot_y) by summing CURVE contributions
instead of calling the hardcoded kernels. Each axis function mirrors its
kernel aggregator (`_dx` / `_dy` / `_z_push` / `_rotation` / `_note_tilt`)
branch for branch, building a curve per active mod and summing it over the
batch's y_offset in one vectorized pass.

Not yet curve-ported, so still delegated to the kernel helpers: the ZOOM
family (mini / tiny / pulse / shrink / confusion-zoom / z-reprojection),
the ALPHA / visibility family (blink / boomerang / hidden-sudden), the
per-note GLOW, and the perspective reprojections (hallway / confusiony).
`assemble` calls `arrow_effects._zoom` / `_alpha` / etc. for those, so the
returned `NoteOffsets` is identical to `note_offsets` while the position
core runs on curves.

Numbered per-column variants (`drunk0`..) fold into a per-note percent
array via `per_note_percent`, which the curve's `scale` broadcasts. The
build is per-frame (as `note_offsets` is), so a mod's percent is closed
over at build; unmodded axes cost nothing (no curve is built).
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves as mc
from analysis.player.render.mods import mod_curves_beatmove as mb
from analysis.player.render.mods import mod_curves_columns as mcol
from analysis.player.render.mods import mod_curves_rotation as mrot
from analysis.player.render.mods import mod_curves_warps as mw
from analysis.player.render.mods import mod_curves_waveform as mv
from analysis.player.render.mods import mod_curves_zsiblings as mz
from analysis.player.render.mods import mod_curves_perspective as mcp
from analysis.player.render.mods import mod_curves_zoom as mzoom
from analysis.player.render.mods import mod_curves_alpha as mca
from analysis.player.render.mods import arrow_effects as ae


def per_note_percent(percents, name, cols, keycount):
    """A mod percent as a per-note array, folding numbered per-column
    variants (`<name>0`..) via `column_percents` (mirrors `_per_note`)."""
    per_col = {c: percents[f'{name}{c}']
               for c in range(keycount) if f'{name}{c}' in percents}
    return ae.column_percents(percents.get(name, 0.0), cols, keycount,
                              per_col or None)


def is_active(percents, name, keycount):
    """Live if the global percent is nonzero or any numbered per-column
    variant is present (mirrors `_active`)."""
    return bool(percents.get(name, 0.0)) or \
        any(f'{name}{c}' in percents for c in range(keycount))


def _g(percents, name):
    return percents.get(name, 0.0)


def _sum(curves, y_offset, ctx):
    if not curves:
        return np.zeros(ctx.cols.shape[0], dtype=np.float64)
    return cv.add(*curves)(y_offset, ctx)


# ---------------------------------------------------------------------------
# Curve builders per aggregator branch. Each mirrors arrow_effects exactly.
# ---------------------------------------------------------------------------

def _drunk_pair(p, cols, kc, size, suffix):
    """drunk + tandrunk for one axis (suffix '' = X, 'z' = Z)."""
    out = []
    for base, is_tan in (('drunk' + suffix, False), ('tandrunk' + suffix, True)):
        if is_active(p, base, kc):
            pct = per_note_percent(p, base, cols, kc)
            out.append(mc.drunk_x(pct, speed=_g(p, base + 'speed'),
                                  offset=_g(p, base + 'offset'),
                                  period=_g(p, base + 'period'), is_tan=is_tan))
    return out


def _tornado_pair(p, cols, kc, size, suffix):
    out = []
    build = mw.tornado_x if suffix == '' else mz.tornadoz_z
    for base, is_tan in (('tornado' + suffix, False),
                         ('tantornado' + suffix, True)):
        if _g(p, base):
            out.append(build(_g(p, base), kc, size,
                             offset=_g(p, base + 'offset'),
                             period=_g(p, base + 'period'), is_tan=is_tan))
    return out


def _warp_family(p, cols, kc, size, suffix, tan_digital):
    """digital / zigzag / sawtooth / square / bounce for one axis."""
    dig, zz, saw, sq, bo = (mv.digital_x, mv.zigzag_x, mv.sawtooth_x,
                            mv.square_x, mv.bounce_x)
    if suffix == 'z':
        dig, zz, saw, sq, bo = (mz.digitalz_z, mz.zigzagz_z, mz.sawtoothz_z,
                                mz.squarez_z, mz.bouncez_z)
    n = lambda base: base + suffix
    per = lambda base: per_note_percent(p, n(base), cols, kc)
    c = lambda base, sub: _g(p, n(base) + sub)
    out = []
    if is_active(p, n('digital'), kc):
        out.append(dig(per('digital'), c('digital', 'offset'),
                       c('digital', 'period'), c('digital', 'steps'), size))
    if is_active(p, n('zigzag'), kc):
        out.append(zz(per('zigzag'), c('zigzag', 'offset'),
                      c('zigzag', 'period'), size))
    if is_active(p, n('sawtooth'), kc):
        out.append(saw(per('sawtooth'), c('sawtooth', 'period'), size))
    if is_active(p, n('square'), kc):
        out.append(sq(per('square'), c('square', 'offset'),
                      c('square', 'period'), size))
    if is_active(p, n('bounce'), kc):
        out.append(bo(per('bounce'), c('bounce', 'offset'),
                      c('bounce', 'period'), size))
    if tan_digital and is_active(p, n('tandigital'), kc):
        out.append(dig(per('tandigital'), c('tandigital', 'offset'),
                       c('tandigital', 'period'), c('tandigital', 'steps'),
                       size, is_tan=True))
    return out


def dx_curves(p, cols, kc, size, beat_now):
    out = _drunk_pair(p, cols, kc, size, '')
    out += _tornado_pair(p, cols, kc, size, '')
    if is_active(p, 'bumpyx', kc):
        out.append(mc.bumpy_z(per_note_percent(p, 'bumpyx', cols, kc),
                              offset=_g(p, 'bumpyxoffset'),
                              period=_g(p, 'bumpyxperiod')))
    if is_active(p, 'tanbumpyx', kc):
        out.append(mc.bumpy_z(per_note_percent(p, 'tanbumpyx', cols, kc),
                              offset=_g(p, 'tanbumpyxoffset'),
                              period=_g(p, 'tanbumpyxperiod'), is_tan=True))
    if _g(p, 'flip'):
        out.append(mcol.flip_x(_g(p, 'flip'), kc, size))
    if _g(p, 'invert'):
        out.append(mcol.invert_x(_g(p, 'invert'), kc, size))
    if is_active(p, 'beat', kc):
        out.append(mb.beat_x(per_note_percent(p, 'beat', cols, kc), beat_now,
                             offset=_g(p, 'beatoffset'),
                             period=_g(p, 'beatperiod'), mult=_g(p, 'beatmult')))
    if _g(p, 'parabolax'):
        out.append(mb.parabola(_g(p, 'parabolax'), size))
    if _g(p, 'attenuatex'):
        out.append(mb.attenuate(_g(p, 'attenuatex'), kc, size))
    if _g(p, 'xmode'):
        out.append(mb.xmode_x(_g(p, 'xmode')))
    if is_active(p, 'movex', kc):
        out.append(mb.movex_x(per_note_percent(p, 'movex', cols, kc), size))
    out += _warp_family(p, cols, kc, size, '', tan_digital=True)
    # Perspective X (hallway recede + confusiony horizontal foreshorten),
    # gated exactly as arrow_effects._perspective_dx.
    if _g(p, 'hallway'):
        out.append(mcp.hallway_x(_g(p, 'hallway'), kc, size))
    if (_g(p, 'confusiony') or _g(p, 'confusionyoffset')
            or is_active(p, 'confusiony', kc)):
        offset = ae._confusion_offset(p, 'confusiony', cols, kc)
        out.append(mcp.confusiony_dx(_g(p, 'confusiony'), offset, kc, size))
    return out


def dy_curves(p, cols, kc, size, beat_now):
    out = []
    if is_active(p, 'tipsy', kc):
        out.append(mw.tipsy_y(per_note_percent(p, 'tipsy', cols, kc),
                              speed=_g(p, 'tipsyspeed'), offset=_g(p, 'tipsyoffset')))
    if _g(p, 'tantipsy'):
        out.append(mw.tipsy_y(_g(p, 'tantipsy'), speed=_g(p, 'tantipsyspeed'),
                              offset=_g(p, 'tantipsyoffset'), is_tan=True))
    if _g(p, 'beaty'):
        out.append(mb.beat_y(_g(p, 'beaty'), beat_now, offset=_g(p, 'beatyoffset'),
                             period=_g(p, 'beatyperiod'), mult=_g(p, 'beatymult')))
    if _g(p, 'parabolay'):
        out.append(mb.parabola(_g(p, 'parabolay'), size))
    if _g(p, 'attenuatey'):
        out.append(mb.attenuate(_g(p, 'attenuatey'), kc, size))
    if is_active(p, 'movey', kc):
        out.append(mb.movey_y(per_note_percent(p, 'movey', cols, kc), size))
    return out


def z_curves(p, cols, kc, size, beat_now):
    out = _warp_family(p, cols, kc, size, 'z', tan_digital=True)
    if _g(p, 'bumpy'):
        out.append(mc.bumpy_z(_g(p, 'bumpy'), offset=_g(p, 'bumpyoffset'),
                              period=_g(p, 'bumpyperiod')))
    if _g(p, 'tanbumpy'):
        out.append(mc.bumpy_z(_g(p, 'tanbumpy'), offset=_g(p, 'tanbumpyoffset'),
                              period=_g(p, 'tanbumpyperiod'), is_tan=True))
    out += _drunk_pair(p, cols, kc, size, 'z')
    out += _tornado_pair(p, cols, kc, size, 'z')
    if _g(p, 'beatz'):
        out.append(mb.beat_z(_g(p, 'beatz'), beat_now, offset=_g(p, 'beatzoffset'),
                             period=_g(p, 'beatzperiod'), mult=_g(p, 'beatzmult')))
    if _g(p, 'parabolaz'):
        out.append(mb.parabola(_g(p, 'parabolaz'), size))
    if _g(p, 'attenuatez'):
        out.append(mb.attenuate(_g(p, 'attenuatez'), kc, size))
    if is_active(p, 'movez', kc):
        out.append(mb.movez_z(per_note_percent(p, 'movez', cols, kc), size))
    return out


def rotation_curves(p, kc):
    out = []
    if _g(p, 'dizzy'):
        out.append(mrot.dizzy_rot(_g(p, 'dizzy')))
    if _g(p, 'confusion') or _g(p, 'confusionoffset'):
        out.append(mrot.confusion_rot(_g(p, 'confusion'), _g(p, 'confusionoffset')))
    return out


def assemble(percents, cols, y_offset, t_now, beat_now, keycount,
             note_beats=None, bps=2.0, arrow_size=ae.ARROW_SIZE,
             rand_seed=0, project_3d=False):
    """A `NoteOffsets` built entirely from composable spatial curves --
    the drop-in `arrow_effects.note_offsets` delegates to. Every axis
    (dx/dy/z/rotation/roll/twirl, zoom, alpha, glow) is assembled from the
    per-family curve builders; byte-equal to the former kernel body
    (tests/test_note_curves.py). Signature mirrors `note_offsets`."""
    cols = np.asarray(cols)
    y_offset = np.asarray(y_offset, dtype=np.float64)
    n = cols.shape[0]
    note_beats = (np.full(n, beat_now, dtype=np.float64) if note_beats is None
                  else np.asarray(note_beats, dtype=np.float64))
    ctx = cv.Ctx(t=t_now, beat=beat_now, cols=cols, note_beats=note_beats,
                 arrow_size=arrow_size)

    dx = _sum(dx_curves(percents, cols, keycount, arrow_size, beat_now),
              y_offset, ctx)
    dy = _sum(dy_curves(percents, cols, keycount, arrow_size, beat_now),
              y_offset, ctx)
    rotation = _sum(rotation_curves(percents, keycount), y_offset, ctx)

    # z_push accumulates every engine +z contribution; the 2D path
    # reprojects it to zoom (waveform_push), the project_3d path hands it
    # to the camera and skips that reprojection (z_push arg).
    z_sum = _sum(z_curves(percents, cols, keycount, arrow_size, beat_now),
                 y_offset, ctx)
    z = rot_x = rot_y = None
    if project_3d:
        z = np.broadcast_to(z_sum, (n,)).astype(np.float64)
        rot_x, rot_y = _note_tilt(percents, y_offset, n)
    zoom = mzoom.zoom_curve(percents, cols, keycount, arrow_size,
                            beat_now=beat_now, z_push=z,
                            waveform_push=z_sum)(y_offset, ctx)

    # Visibility samples the raw scroll offset plus TIPSY only (the sole dy
    # mod that moves a note through the hidden/sudden windows).
    vis_y = y_offset + _sum(_tipsy_curves(percents, cols, kc=keycount,
                                          size=arrow_size), y_offset, ctx)
    alpha = mca.alpha_curve(percents, t_now)(vis_y, ctx)
    glow = None
    if is_active(percents, 'stealthglow', keycount):
        glow = mca.glow_curve(
            ae.column_add(percents, 'stealthglow', cols),
            past_receptors=bool(_g(percents, 'stealthpastreceptors')))(
                vis_y, ctx)

    return ae.NoteOffsets(dx=dx, dy=dy, rotation_deg=rotation,
                          alpha_mult=alpha, zoom=zoom, z=z, rot_x=rot_x,
                          rot_y=rot_y, glow=glow)


def _note_tilt(percents, y_offset, n):
    """(rot_x, rot_y) out-of-plane tilt in degrees: roll -> RotationX,
    twirl -> RotationY (GetRotationX/Y = effect * yOffset/2), rest 0."""
    roll, twirl = _g(percents, 'roll'), _g(percents, 'twirl')
    y = np.asarray(y_offset, dtype=np.float64)
    rot_x = roll * y / 2.0 if roll else np.zeros(n)
    rot_y = twirl * y / 2.0 if twirl else np.zeros(n)
    return (np.broadcast_to(rot_x, (n,)).astype(np.float64),
            np.broadcast_to(rot_y, (n,)).astype(np.float64))


def _tipsy_curves(percents, cols, kc, size):
    """The tipsy + tantipsy dy alone (the visibility-affecting dy), for
    vis_y. A subset of dy_curves."""
    out = []
    if is_active(percents, 'tipsy', kc):
        out.append(mw.tipsy_y(per_note_percent(percents, 'tipsy', cols, kc),
                              speed=_g(percents, 'tipsyspeed'),
                              offset=_g(percents, 'tipsyoffset')))
    if _g(percents, 'tantipsy'):
        out.append(mw.tipsy_y(_g(percents, 'tantipsy'),
                              speed=_g(percents, 'tantipsyspeed'),
                              offset=_g(percents, 'tantipsyoffset'), is_tan=True))
    return out
