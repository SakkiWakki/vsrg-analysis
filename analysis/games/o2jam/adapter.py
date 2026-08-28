"""O2Jam game adapter + scroll mode.

O2Jam has no replay format to load, so this adapter exists for its scroll
mode: it lets any loaded replay be read at an O2Jam speed and lets O2Jam
players translate their /csp value to and from the other games.
"""
from __future__ import annotations

from analysis.core.game import GameAdapter
from analysis.player import scroll


class O2JamAdapter(GameAdapter):
    name = 'o2jam'

    def default_scroll_mode(self) -> str:
        return 'csp'


# --- O2Jam scroll mode -------------------------------------------------------
# The Constant Speed Patch replaces the note speed multiplier (a float the
# client keeps at 1.0 for x1) with `240 / round(240 * BPM / CSP)`, which is
# CSP/BPM quantised to 240/n. The BPM cancels out of BPM * multiplier, so a
# CSP value *is* the scroll rate: the x86 the launcher patches in at
# OTwo.exe+844897 does exactly this, and Commands.txt agrees that BPM 150 at
# /csp 600 is 4x.
#
# One measure spans the 480px noteplain at x1, so px/s = 2 * CSP. That is
# the only reading of the pixel side that puts the usable x2-x6 range at
# sane travel times (0.8s down to 0.27s), but unlike the multiplier algebra
# above it is not proven from the binary.
_O2JAM_PX_PER_S_PER_CSP = 2.0


def _o2jam_pxps_at_reference_field(value):
    return value * _O2JAM_PX_PER_S_PER_CSP


def _o2jam_to_pxps(value, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return _o2jam_pxps_at_reference_field(float(value)) * field_scale


def _o2jam_from_pxps(pxps, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return pxps / (_o2jam_pxps_at_reference_field(1.0) * field_scale)


scroll.register(scroll.ScrollMode(
    key='csp',
    label='O2Jam CSP',
    game='o2jam',
    to_pxps=_o2jam_to_pxps,
    from_pxps=_o2jam_from_pxps,
    default_value=600.0,
    value_bounds=(60.0, 2000.0),
    nudge=scroll.integer_step_nudge,
    format_value=lambda v: (f'csp {int(v)}' if abs(v - round(v)) < 1e-4
                            else f'csp {v:.1f}'),
))


ADAPTER = O2JamAdapter()
