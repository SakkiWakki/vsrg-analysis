"""Parity-fixture generator for src/ease.rs / src/channels.rs eased
sampling (agent G2, wave 7, item C12).

Drives the REAL Python easing sampler - `effects.timeline.EventTimeline`
over a single eased Keyframe - across the whole ease vocabulary the repo
supports (osu.Framework ids 0..=23 plus the StepMania negative ids) and
dumps the sampled value on a t-grid. The Rust test builds the equivalent
one-ramp channel via `ChannelTable::push_eased` and asserts the ported
curve family reproduces the same values, so a Python-compiled keyframe
crosses Seam A losslessly (no 1/30s densification needed).

Run: PYTHONPATH=/home/yucky/dev/vsrg-analysis \
    python analysis/player/render/storyboard/native/gen_fixtures/eases.py

Each case is one channel with breakpoints
  ts=[t0, t1] vals=[v0, v1] durs=[dur, 0] eases=[easing, 0] rest=v0
sampled at a grid of times; EventTimeline plays back exactly this shape
(rest before t0 - here v0, eased ramp over [t0, t1], hold after).
"""
from __future__ import annotations

import json
import pathlib

from analysis.player.render.effects.easing import (
    EASE_SM_BOUNCE_BEGIN, EASE_SM_BOUNCE_END, EASE_SM_SPRING,
)
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

# The full osu.Framework enum range the port implements plus its exotic
# tail (which both Python and Rust fold to OutQuint), and the SM curves.
_EASE_IDS = list(range(0, 24)) + [
    30, 99,                       # exotic-tail ids -> OutQuint fallback
    EASE_SM_BOUNCE_BEGIN, EASE_SM_BOUNCE_END, EASE_SM_SPRING,
]

# One ramp shape reused per ease id: v0 -> v1 over [t0, t0+dur].
_T0, _DUR, _V0, _V1 = 2.0, 4.0, -5.0, 11.0

# Sample densely across the ramp plus the flat tails on either side.
_GRID = [round(_T0 - 1.0 + 0.1 * i, 4) for i in range(0, 61)]


def _case(easing: int) -> dict:
    kf = Keyframe(t=_T0, values=(_V1,), duration=_DUR, easing=easing, start=(_V0,))
    tl = EventTimeline([kf], rest=(_V0,))
    samples = [tl.sample(t)[0] for t in _GRID]
    return {
        'easing': easing,
        't0': _T0,
        'dur': _DUR,
        'v0': _V0,
        'v1': _V1,
        'grid': _GRID,
        'expected': samples,
    }


def main() -> None:
    cases = [_case(e) for e in _EASE_IDS]
    out = pathlib.Path(__file__).resolve().parent.parent / 'fixtures'
    out.mkdir(exist_ok=True)
    path = out / 'ease_cases.json'
    path.write_text(json.dumps({'cases': cases}, indent=1))
    print(f'wrote {len(cases)} ease cases x {len(_GRID)} samples -> {path}')


if __name__ == '__main__':
    main()
