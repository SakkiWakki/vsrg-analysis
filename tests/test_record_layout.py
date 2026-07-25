"""The Python record mirror must match what the Rust evaluator emits.

`record.py` restates `native/src/evaluate.rs`'s op-stream layout because
Python cannot read the Rust constants. That mirror is the failure mode this
file exists to catch: a lane added in Rust without updating it makes every
reader index the wrong column, silently, with no build error anywhere.
"""
import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')

from analysis.player.render.storyboard import record


def _one_blit_evaluator():
    builder = sn.DocBuilder(640.0, 480.0)
    builder.item(0, sn.SRC_FILL, 0, x_rest=10.0, y_rest=20.0)
    return builder.finish()


def test_strides_match_the_evaluator():
    evaluator = _one_blit_evaluator()
    assert record.U_STRIDE == evaluator.u_stride
    assert record.F_STRIDE == evaluator.f_stride


def test_every_f_lane_offset_is_inside_the_stride():
    # An offset past the end reads another op's row (numpy reshapes without
    # complaint), so this is not merely a tidiness check.
    named = {name: value for name, value in vars(record).items()
             if name.startswith('F_') and name != 'F_STRIDE'}
    assert named, 'expected F_* lane offsets'
    for name, offset in named.items():
        assert 0 <= offset < record.F_STRIDE, f'{name}={offset}'


def test_every_u_lane_offset_is_inside_the_stride():
    named = {name: value for name, value in vars(record).items()
             if name.startswith('U_') and name != 'U_STRIDE'}
    assert named, 'expected U_* lane offsets'
    for name, offset in named.items():
        assert 0 <= offset < record.U_STRIDE, f'{name}={offset}'


def test_op_and_source_codes_match_the_extension():
    assert record.OP_BLIT == sn.OP_BLIT
    assert record.OP_COPY == sn.OP_COPY
    assert record.SRC_IMAGE == sn.SRC_IMAGE
    assert record.SRC_DRAWABLE == sn.SRC_DRAWABLE
    assert record.SRC_FILL == sn.SRC_FILL


def test_the_named_lanes_carry_what_they_claim():
    # Reads the lanes back off a real record, so a renumbering that keeps
    # every offset in range is still caught.
    evaluator = _one_blit_evaluator()
    u_bytes, f_bytes, _uf, n = evaluator.frame(0.0)
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, record.U_STRIDE)
    f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, record.F_STRIDE)
    row = next(i for i in range(n) if u[i, record.U_KIND] == record.OP_BLIT)

    assert u[row, record.U_A] == record.SRC_FILL
    assert f[row, record.F_MAT + 2] == 10.0    # mat3 tx
    assert f[row, record.F_MAT + 5] == 20.0    # mat3 ty
    assert f[row, record.F_OPACITY] == 1.0
    assert tuple(f[row, record.F_TINT:record.F_TINT + 3]) == (1.0, 1.0, 1.0)
    assert tuple(f[row, record.F_SIZE:record.F_SIZE + 2]) == (
        record.SIZE_NATURAL, record.SIZE_NATURAL)
    assert float(f[row, record.F_FIT]) < record.FIT_OFF_BELOW
    assert not any(f[row, record.F_FADE:record.F_FADE + 4])
