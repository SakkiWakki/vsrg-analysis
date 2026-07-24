"""Generate schedule_cases.json: drive the REAL Python lower() over a
case grid and dump (schedule tree, t0, horizon, initial state, expected
lowered breakpoint runs + fire times) for the Rust parity test.

Run with:
    PYTHONPATH=/home/yucky/dev/vsrg-analysis \
        python analysis/player/render/storyboard/native/gen_fixtures/schedule.py

The Rust port targets `LoweredProp { ts, vals, durs }` - the
ChannelTable::push breakpoint triple (dur 0 = hold). This script converts
lower()'s Ramp/Hold emissions into that triple with the SAME coalescing
rule the Rust `to_props` uses, so parity is Rust-lower vs Python-lower over
one shared emission->breakpoint mapping. Property keys are u32 (the frozen
signature), so schedules here name props by integer id.
"""
from __future__ import annotations

import json
import os

from analysis.player.render.schedule import (
    Add, Hibernate, Hold, Loop, Ramp, Seg, Seq, lower,
)

# Opaque effect sentinel: any non-schedule object emits a Fire. We tag
# cases that use one with fire_id >= 0 in the JSON; the object identity is
# irrelevant to lower() (it only records the time), so a plain marker works.
_FIRE = object()


def node_json(node):
    """Serialize a schedule node to the JSON the Rust `build` reads."""
    if isinstance(node, Seg):
        targets = []
        for prop, dest in node.targets.items():
            if isinstance(dest, Add):
                targets.append({'prop': prop, 'mode': 'add', 'value': dest.delta})
            else:
                targets.append({'prop': prop, 'mode': 'abs', 'value': float(dest)})
        out = {'kind': 'Seg', 'dur': node.dur, 'ease': node.ease, 'targets': targets}
        if isinstance(node.effect, (Seg, Seq, Hibernate, Loop)):
            out['effect'] = node_json(node.effect)
            out['fire_id'] = -1
        elif node.effect is not None:
            out['effect'] = None
            out['fire_id'] = 0
        else:
            out['effect'] = None
            out['fire_id'] = -1
        return out
    if isinstance(node, Seq):
        return {'kind': 'Seq', 'parts': [node_json(p) for p in node.parts]}
    if isinstance(node, Hibernate):
        return {'kind': 'Hibernate', 'dur': node.dur}
    if isinstance(node, Loop):
        return {'kind': 'Loop', 'period': node.period, 'body': node_json(node.body)}
    raise TypeError(node)


def emissions_to_props(emissions):
    """Mirror the Rust `to_props`: interleaved Ramp/Hold -> per-prop
    (ts, vals, durs), first-touched prop order, coalescing a terminal hold
    that a following ramp starts on."""
    order = []
    props = {}

    def push_bp(lp, t, v, dur):
        ts, vals, durs = lp
        if ts and ts[-1] == t and vals[-1] == v and durs[-1] == 0.0:
            durs[-1] = dur
            return
        ts.append(t)
        vals.append(v)
        durs.append(dur)

    for e in emissions:
        prop = e.prop
        if prop not in props:
            order.append(prop)
            props[prop] = ([], [], [])
        lp = props[prop]
        if isinstance(e, Ramp):
            push_bp(lp, e.t0, e.v0, max(0.0, e.t1 - e.t0))
            push_bp(lp, e.t1, e.v1, 0.0)
        elif isinstance(e, Hold):
            push_bp(lp, e.t, e.v, 0.0)

    return [
        {'prop': p, 'ts': props[p][0], 'vals': props[p][1], 'durs': props[p][2]}
        for p in order
    ]


def case(name, node, *, t0=0.0, horizon=None, state=None):
    st = dict(state or {})
    lowered = lower(node, t0=t0, state=st, until=horizon)
    return {
        'name': name,
        'node': node_json(node),
        't0': t0,
        'horizon': horizon,
        'state': [[int(k), float(v)] for k, v in (state or {}).items()],
        'expected_props': emissions_to_props(lowered.emissions),
        'expected_fires': [f.t for f in lowered.fires],
    }


# Prop id conventions used below (arbitrary but stable integers).
X, Y, ROT, ZOOM, ALPHA = 0, 1, 2, 3, 4


def build_cases():
    cases = []

    # 1. Single ramp from seeded state.
    cases.append(case('single_ramp',
                      Seg(dur=1.0, targets={X: 100.0}), state={X: 0.0}))

    # 2. Ramp with an unseeded prop (v0 defaults to 0).
    cases.append(case('ramp_unseeded', Seg(dur=2.0, targets={Y: 50.0})))

    # 3. Zero-duration write -> Hold.
    cases.append(case('zero_dur_hold',
                      Seg(dur=0.0, targets={X: 42.0}), state={X: 1.0}))

    # 4. Unchanged target emits nothing.
    cases.append(case('unchanged_noop',
                      Seg(dur=1.0, targets={X: 5.0}), state={X: 5.0}))

    # 5. Ease id carried (nonlinear ease number preserved in breakpoints
    #    only via linear dur; ease affects sampling not breakpoints, so this
    #    is a plain ramp for the triple form).
    cases.append(case('ramp_ease',
                      Seg(dur=1.5, ease=2, targets={ROT: 90.0}), state={ROT: 0.0}))

    # 6. Sequence of two ramps chaining on the same prop.
    cases.append(case('seq_two_ramps',
                      Seq(Seg(dur=1.0, targets={X: 100.0}),
                          Seg(dur=1.0, targets={X: 0.0})),
                      state={X: 0.0}))

    # 7. Sequence with a hibernate prefix pushing the second ramp later.
    cases.append(case('seq_hibernate_between',
                      Seq(Seg(dur=1.0, targets={X: 10.0}),
                          Hibernate(dur=2.0),
                          Seg(dur=1.0, targets={X: 20.0})),
                      state={X: 0.0}))

    # 8. Leading hibernate carries the whole schedule forward.
    cases.append(case('leading_hibernate',
                      Seq(Hibernate(dur=3.0),
                          Seg(dur=1.0, targets={Y: 5.0})),
                      state={Y: 0.0}))

    # 9. Multiple props in one seg (order preserved).
    cases.append(case('multi_prop_seg',
                      Seg(dur=1.0, targets={X: 10.0, Y: 20.0, ROT: 30.0}),
                      state={X: 0.0, Y: 0.0, ROT: 0.0}))

    # 10. Relative Add onto seeded state.
    cases.append(case('add_relative',
                      Seg(dur=1.0, targets={X: Add(15.0)}), state={X: 5.0}))

    # 11. Add onto unseeded (base 0).
    cases.append(case('add_unseeded', Seg(dur=1.0, targets={ROT: Add(45.0)})))

    # 12. Add zero-delta from seeded == unchanged -> noop.
    cases.append(case('add_zero_noop',
                      Seg(dur=1.0, targets={X: Add(0.0)}), state={X: 7.0}))

    # 13. Chained adds accumulate.
    cases.append(case('chained_adds',
                      Seq(Seg(dur=1.0, targets={X: Add(10.0)}),
                          Seg(dur=1.0, targets={X: Add(10.0)}),
                          Seg(dur=1.0, targets={X: Add(-5.0)})),
                      state={X: 0.0}))

    # 14. Opaque effect fires at segment start.
    cases.append(case('opaque_fire',
                      Seg(dur=1.0, targets={X: 5.0}, effect=_FIRE),
                      state={X: 0.0}))

    # 15. Opaque fire on a zero-dur command (pure fire, no emission).
    cases.append(case('command_fire_only',
                      Seg(dur=0.0, effect=_FIRE)))

    # 16. Nested SCHEDULE effect joins the queue tail.
    cases.append(case('nested_schedule_tail',
                      Seq(Seg(dur=1.0, targets={X: 10.0},
                              effect=Seg(dur=1.0, targets={Y: 99.0})),
                          Seg(dur=1.0, targets={X: 20.0})),
                      state={X: 0.0, Y: 0.0}))

    # 17. Nested schedule that itself fires an opaque effect.
    cases.append(case('nested_with_fire',
                      Seg(dur=2.0, targets={X: 5.0},
                          effect=Seq(Hibernate(dur=1.0),
                                     Seg(dur=0.0, effect=_FIRE))),
                      state={X: 0.0}))

    # 18. Simple loop unrolled to horizon.
    cases.append(case('loop_unroll',
                      Loop(period=2.0, body=Seg(dur=1.0, targets={X: 10.0})),
                      horizon=7.0, state={X: 0.0}))

    # 19. Loop whose body toggles a prop (ping-pong).
    cases.append(case('loop_pingpong',
                      Loop(period=2.0,
                           body=Seq(Seg(dur=1.0, targets={X: 100.0}),
                                    Seg(dur=1.0, targets={X: 0.0}))),
                      horizon=6.5, state={X: 0.0}))

    # 20. Loop with body shorter than period (hibernate fill).
    cases.append(case('loop_period_fill',
                      Loop(period=3.0, body=Seg(dur=1.0, targets={ROT: Add(90.0)})),
                      horizon=8.0, state={ROT: 0.0}))

    # 21. Loop preceded by a lead-in seg.
    cases.append(case('loop_after_leadin',
                      Seq(Seg(dur=1.0, targets={X: 5.0}),
                          Loop(period=2.0, body=Seg(dur=1.0, targets={X: Add(1.0)}))),
                      horizon=6.0, state={X: 0.0}))

    # 22. Nonzero t0 offset shifts everything.
    cases.append(case('t0_offset',
                      Seg(dur=1.0, targets={X: 10.0}), t0=5.0, state={X: 0.0}))

    # 23. Horizon truncates a long sequence mid-fold.
    cases.append(case('horizon_truncates',
                      Seq(Seg(dur=1.0, targets={X: 10.0}),
                          Seg(dur=1.0, targets={X: 20.0}),
                          Seg(dur=1.0, targets={X: 30.0})),
                      horizon=1.5, state={X: 0.0}))

    # 24. Negative dur clamps to 0 (max(0, dur)).
    cases.append(case('negative_dur_clamp',
                      Seg(dur=-3.0, targets={X: 9.0}), state={X: 0.0}))

    # 25. Negative hibernate clamps to 0.
    cases.append(case('negative_hibernate_clamp',
                      Seq(Hibernate(dur=-2.0),
                          Seg(dur=1.0, targets={X: 4.0})),
                      state={X: 0.0}))

    # 26. Deep sequence exercising many chained props + adds.
    cases.append(case('deep_mixed',
                      Seq(Seg(dur=0.5, targets={X: 1.0, Y: 2.0}),
                          Seg(dur=0.0, targets={ROT: 45.0}),
                          Hibernate(dur=0.25),
                          Seg(dur=1.0, targets={X: Add(3.0), ZOOM: 2.0}),
                          Seg(dur=1.0, targets={ALPHA: 0.0}, effect=_FIRE)),
                      state={X: 0.0, Y: 0.0, ROT: 0.0, ZOOM: 1.0, ALPHA: 1.0}))

    # 27. Multiple opaque fires in a sequence (fire time order).
    cases.append(case('multi_fire_order',
                      Seq(Seg(dur=1.0, effect=_FIRE),
                          Seg(dur=1.0, effect=_FIRE),
                          Seg(dur=0.0, effect=_FIRE)),
                      state={}))

    # 28. Loop body with a nested opaque fire each pass.
    cases.append(case('loop_fires_each_pass',
                      Loop(period=2.0,
                           body=Seg(dur=0.0, effect=_FIRE)),
                      horizon=5.0))

    # 29. Zero-dur re-write of same value -> noop after first hold.
    cases.append(case('repeat_hold_same_value',
                      Seq(Seg(dur=0.0, targets={X: 3.0}),
                          Seg(dur=0.0, targets={X: 3.0})),
                      state={X: 0.0}))

    # 30. Ramp then hold-same (terminal value re-asserted) coalesces.
    cases.append(case('ramp_then_hold_same',
                      Seq(Seg(dur=1.0, targets={X: 10.0}),
                          Seg(dur=0.0, targets={X: 10.0})),
                      state={X: 0.0}))

    return cases


def main():
    cases = build_cases()
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
    out_path = os.path.abspath(os.path.join(out_dir, 'schedule_cases.json'))
    with open(out_path, 'w') as f:
        json.dump(cases, f, indent=1)
        f.write('\n')
    print(f'wrote {len(cases)} cases -> {out_path}')


if __name__ == '__main__':
    main()
