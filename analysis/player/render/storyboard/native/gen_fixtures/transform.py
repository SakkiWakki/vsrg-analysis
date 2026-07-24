"""Parity-fixture generator for src/transform.rs (agent A1, wave 1).

Drives the REAL Python `field_compose.TransformChannel` over a synthetic
case grid - anchors x flip_base_y x crops x rotation x multi-link chains
x hidden - and dumps each case's per-link sampled scalars plus the
expected outputs (H as 9 row-major floats, alpha, crop) to
fixtures/transform_cases.json. The Rust test reads it via include_str!
and asserts the pure-math port reproduces the same 2D composition.

Run: PYTHONPATH=/home/yucky/dev/vsrg-analysis python gen_fixtures/transform.py

Scope note (matches the frozen A1 signature): this covers the leaf-link
2D affine path only - z=0, centered fov default, no rotation_x/y, no
quat, no skew-before toggle. Those are the A3 camera area's perspective
math; here the design projection reduces the z=0 plane 1:1 to design px,
so `at()`'s normalized homography is exactly the affine block of the
composed `_TO_CONTENT @ world`.
"""
from __future__ import annotations

import json
import pathlib

from analysis.games.notitg import field_compose
from analysis.player.render.effects.timeline import Keyframe

# Every 2D scalar the Rust TransformState carries, with its SM rest.
# (`awake` and the out-of-plane axes stay at rest here: they are outside
# this file's 2D-only port surface - see the module docstring.)
_STATE_FIELDS = {
    'x': 0.0, 'y': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0,
    'rotation': 0.0, 'skew_x': 0.0, 'skew_y': 0.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0,
    'halign': 0.5, 'valign': 0.5,
    'hidden': 0.0, 'alpha': 1.0,
    'crop_left': 0.0, 'crop_top': 0.0, 'crop_right': 0.0, 'crop_bottom': 0.0,
}


def _link(state):
    """A field_compose link timeline dict from one scalar state (each
    property an immediate keyframe at t=0, so sampling at t=1 holds it)."""
    keyframes = {name: [Keyframe(0.0, (value,), 0.0, 0)]
                 for name, value in state.items()}
    return field_compose.link_timelines(keyframes)


def _fill(**overrides):
    state = dict(_STATE_FIELDS)
    state.update(overrides)
    return state


def _case(name, link_states, flip_base_y):
    tc = field_compose.TransformChannel([_link(s) for s in link_states],
                                        flip_base_y=flip_base_y)
    result = tc.at(1.0)
    if result is None:
        expected = None
    else:
        homography, alpha = result
        crop = tc.crop_at(1.0)
        expected = {
            'h': [float(v) for v in homography.flatten()],
            'alpha': float(alpha),
            'crop': list(crop) if crop is not None else None,
        }
    return {
        'name': name,
        'flip_base_y': flip_base_y,
        'links': [{k: float(v) for k, v in s.items()} for s in link_states],
        'expected': expected,
    }


def _cases():
    cases = []

    # --- single-link anchors (halign / valign grid) ---------------------
    for ha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for va in (0.0, 0.5, 1.0):
            cases.append(_case(f'anchor_{ha}_{va}',
                               [_fill(halign=ha, valign=va)], False))

    # --- position + scale + rotation singles ----------------------------
    cases.append(_case('translate', [_fill(x=50.0, y=-30.0)], False))
    cases.append(_case('scale', [_fill(scale_x=1.5, scale_y=0.6)], False))
    cases.append(_case('rotate', [_fill(rotation=37.0)], False))
    cases.append(_case('rotate_neg', [_fill(rotation=-115.0)], False))
    cases.append(_case('scale_rotate_pos',
                       [_fill(scale_x=2.0, scale_y=0.5, rotation=45.0,
                              x=100.0, y=60.0)], False))
    cases.append(_case('skew_x', [_fill(skew_x=0.35)], False))
    cases.append(_case('skew_y', [_fill(skew_y=-0.4)], False))
    cases.append(_case('skew_both',
                       [_fill(skew_x=0.2, skew_y=0.3, rotation=15.0)], False))
    cases.append(_case('base_scale',
                       [_fill(base_scale_x=1.2, base_scale_y=0.9,
                              scale_x=1.1)], False))

    # --- anchor combined with rotation/scale (anchor rides the block) ---
    cases.append(_case('anchor_rotate',
                       [_fill(halign=0.0, valign=1.0, rotation=90.0)], False))
    cases.append(_case('anchor_scale_rotate',
                       [_fill(halign=0.25, valign=0.75, scale_x=1.4,
                              scale_y=0.7, rotation=20.0, skew_x=0.15)], False))

    # --- flip_base_y (single leaf): mirror negates anchor y + swaps crop -
    cases.append(_case('flip_plain', [_fill()], True))
    cases.append(_case('flip_anchor',
                       [_fill(halign=0.25, valign=0.7)], True))
    cases.append(_case('flip_base_scale_cancel',
                       [_fill(base_scale_y=1.0, scale_y=1.0, valign=0.8)], True))
    cases.append(_case('flip_base_scale_neg',
                       [_fill(base_scale_y=-1.0, valign=0.3)], True))
    cases.append(_case('flip_rotate_scale',
                       [_fill(rotation=25.0, scale_x=1.3, scale_y=0.8,
                              valign=0.2, x=40.0)], True))

    # --- crop passthrough (upright) + crop-swap under flip --------------
    cases.append(_case('crop_all', [_fill(crop_left=0.1, crop_top=0.2,
                                          crop_right=0.05, crop_bottom=0.3)],
                       False))
    cases.append(_case('crop_flip_swap',
                       [_fill(crop_top=0.1, crop_bottom=0.35)], True))
    cases.append(_case('crop_rest_upright', [_fill(crop_left=0.0)], False))
    cases.append(_case('crop_lr_only',
                       [_fill(crop_left=0.15, crop_right=0.25)], True))

    # --- multi-link chains (root frame + leaf sprite) -------------------
    cases.append(_case('chain_translate',
                       [_fill(x=320.0, y=200.0), _fill(x=-20.0, y=15.0)],
                       False))
    cases.append(_case('chain_rotate_leaf',
                       [_fill(x=320.0, y=240.0),
                        _fill(rotation=30.0, scale_x=1.2)], False))
    cases.append(_case('chain_rotate_root',
                       [_fill(x=200.0, y=200.0, rotation=45.0),
                        _fill(scale_x=0.8, scale_y=0.8)], False))
    cases.append(_case('chain_anchor_leaf',
                       [_fill(x=320.0, y=240.0, scale_x=1.5),
                        _fill(halign=0.0, valign=0.0, rotation=10.0)], False))
    cases.append(_case('chain_three',
                       [_fill(x=100.0, y=80.0),
                        _fill(rotation=20.0, scale_x=1.1),
                        _fill(scale_y=0.7, skew_x=0.2, halign=0.3)], False))
    cases.append(_case('chain_flip_leaf',
                       [_fill(x=320.0, y=240.0, rotation=15.0),
                        _fill(valign=0.7, base_scale_y=1.0,
                              crop_top=0.1, crop_bottom=0.4)], True))
    cases.append(_case('chain_scale_stack',
                       [_fill(scale_x=2.0, scale_y=2.0, x=160.0),
                        _fill(scale_x=0.5, scale_y=0.5)], False))

    # --- alpha propagation (multiply down the chain) --------------------
    cases.append(_case('alpha_leaf', [_fill(alpha=0.5)], False))
    cases.append(_case('alpha_chain',
                       [_fill(alpha=0.8), _fill(alpha=0.5)], False))
    cases.append(_case('alpha_near_zero', [_fill(alpha=0.001)], False))

    # --- hidden gating (-> None) at leaf and at root -------------------
    cases.append(_case('hidden_leaf', [_fill(hidden=1.0)], False))
    cases.append(_case('hidden_root',
                       [_fill(hidden=1.0), _fill()], False))
    cases.append(_case('hidden_mid',
                       [_fill(), _fill(hidden=1.0), _fill()], False))

    # --- degenerate scale (-> None via near-singular det) --------------
    cases.append(_case('degenerate_scale_x', [_fill(scale_x=0.0)], False))
    cases.append(_case('degenerate_scale_y', [_fill(scale_y=0.0)], False))

    return cases


def main():
    cases = _cases()
    out = pathlib.Path(__file__).resolve().parents[1] / 'fixtures'
    out.mkdir(exist_ok=True)
    path = out / 'transform_cases.json'
    path.write_text(json.dumps({'cases': cases}, indent=1))
    non_null = sum(1 for c in cases if c['expected'] is not None)
    print(f'wrote {len(cases)} cases ({non_null} visible, '
          f'{len(cases) - non_null} None) -> {path}')


if __name__ == '__main__':
    main()
