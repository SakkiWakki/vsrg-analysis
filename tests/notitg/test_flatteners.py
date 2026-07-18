"""Flattener registry, affine matcher, and the kernel==recurrence proof.

The correctness core is `test_kernel_equals_recurrence`: the closed-form kernel
must equal stepping `x = a*x + b` n times from h0, for n = 0..20 across a,b
samples. If the closed form diverges from the scan the SSM path is wrong.

`frame_ir` (Lane 1's file) may not be landed in this worktree yet, so a minimal
stub matching the brief's FROZEN shapes is injected into sys.modules before
`flatteners` imports it. When Lane 1's real module is present it is used as-is;
integration is then a no-op (identical frozen shapes).
"""
import sys
from dataclasses import dataclass, field

import pytest

from analysis.player.render.expr import ast
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.expr.surface import ConstSurface


# -- frame_ir: use Lane 1's if present, else inject the frozen-shape stub -----

try:
    from analysis.games.notitg.frame_ir import ClosedForm, Frame, VarUpdate
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class ClosedForm:
        kernel: object
        coeffs: dict
        h0: object

    @dataclass
    class VarUpdate:
        name: str
        node: object
        closed_form: object = None

    @dataclass
    class Frame:
        window: tuple | None = None
        bindings: dict = field(default_factory=dict)
        children: list = field(default_factory=list)
        parent: object = None

    _stub = type(sys)('analysis.games.notitg.frame_ir')
    _stub.ClosedForm = ClosedForm
    _stub.VarUpdate = VarUpdate
    _stub.Frame = Frame
    sys.modules['analysis.games.notitg.frame_ir'] = _stub

from analysis.games.notitg import flatteners
from analysis.games.notitg.flatteners import (
    AffineFlattener, affine_kernel, route)


# -- helpers -----------------------------------------------------------------


def _update(src: str) -> VarUpdate:
    """One statement of NotITG-Lua source -> a VarUpdate carrying its AST."""
    stmts, _sink = parse_body(src)
    node = stmts[0]
    return VarUpdate(name=src, node=node)


def _empty_frame() -> Frame:
    return Frame(window=None)


def _iterate(a: float, b: float, h0: float, n: int) -> float:
    """Step x = a*x + b exactly n times from h0 - the recurrence itself."""
    x = h0
    for _ in range(n):
        x = a * x + b
    return x


def _coeffs(a: float, b: float) -> dict:
    return {'a': flatteners._const_channel(a), 'b': flatteners._const_channel(b)}


# -- the correctness core: kernel == recurrence ------------------------------

_AB_SAMPLES = [
    (-1.0, 0.0), (-1.0, 3.0), (1.0, 0.0), (1.0, 3.0),
    (0.5, 0.0), (0.5, 3.0), (0.0, 3.0), (2.0, -1.0), (-0.5, 7.0),
]


@pytest.mark.parametrize('a,b', _AB_SAMPLES)
@pytest.mark.parametrize('h0', [0.0, 1.0, -4.0, 2.5])
def test_kernel_equals_recurrence(a, b, h0):
    coeffs = _coeffs(a, b)
    for n in range(21):
        closed = affine_kernel(n, h0, coeffs)
        stepped = _iterate(a, b, h0, n)
        assert closed == pytest.approx(stepped, rel=1e-9, abs=1e-9), (
            f'a={a} b={b} h0={h0} n={n}: kernel {closed} != scan {stepped}')


def test_kernel_toggle_and_step_and_pure_curve():
    # a=-1 wired through the general a!=1 branch: even n -> h0, odd n -> -h0+b.
    toggle = _coeffs(-1.0, 0.0)
    assert affine_kernel(0, 5.0, toggle) == 5.0
    assert affine_kernel(1, 5.0, toggle) == -5.0
    assert affine_kernel(2, 5.0, toggle) == 5.0
    # a=1 sum.
    assert affine_kernel(4, 10.0, _coeffs(1.0, 2.0)) == 18.0
    # a=0 pure curve: b for n>=1 (and h0 at n=0).
    curve = _coeffs(0.0, 7.0)
    assert affine_kernel(0, 7.0, curve) == 7.0
    assert affine_kernel(3, 7.0, curve) == 7.0


# -- AffineFlattener match: the degenerate forms -----------------------------


def _coeff_at0(cf: ClosedForm, name: str) -> float:
    return cf.coeffs[name].at(0.0)


def test_toggle_matches_ssm_a_minus_one():
    cf = AffineFlattener().match(_update('x = x*-1'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == -1.0
    assert _coeff_at0(cf, 'b') == 0.0
    assert cf.kernel is affine_kernel


def test_sum_matches_ssm_a_one_b_step():
    cf = AffineFlattener().match(_update('x = x + 3'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 1.0
    assert _coeff_at0(cf, 'b') == 3.0


def test_sum_step_on_left_side():
    cf = AffineFlattener().match(_update('x = 3 + x'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 1.0
    assert _coeff_at0(cf, 'b') == 3.0


def test_subtract_step():
    cf = AffineFlattener().match(_update('x = x - 2'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 1.0
    assert _coeff_at0(cf, 'b') == -2.0


def test_reflect_subtract_target_on_right():
    # x = 10 - x  ->  a=-1, b=10.
    cf = AffineFlattener().match(_update('x = 10 - x'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == -1.0
    assert _coeff_at0(cf, 'b') == 10.0


def test_geometric_plus_const_matches_ssm():
    cf = AffineFlattener().match(_update('x = 0.5*x + 3'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 0.5
    assert _coeff_at0(cf, 'b') == 3.0


def test_scale_target_on_right():
    cf = AffineFlattener().match(_update('x = 2*x'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 2.0
    assert _coeff_at0(cf, 'b') == 0.0


def test_pure_curve_poke_matches_ssm_a_zero():
    # A poke whose arg does not read the target: the a=0 degenerate. The arg is
    # a compiled constant here (ConstSurface resolves no live symbols).
    cf = AffineFlattener().match(_update('P1:rotationz(45)'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'a') == 0.0
    assert _coeff_at0(cf, 'b') == 45.0
    assert cf.h0 == 45.0     # pure-curve h0 is b(0)


def test_pure_curve_with_const_surface_expression():
    # b compiles through the surface: rotationz(k*2) with k a constant.
    surface = ConstSurface({'k': 20})
    cf = AffineFlattener(surface).match(
        _update('P1:rotationz(k*2)'), _empty_frame())
    assert cf is not None
    assert _coeff_at0(cf, 'b') == 40.0


def test_h0_from_frame_binding():
    frame = Frame(window=None, bindings={'x': 9.0})
    cf = AffineFlattener().match(_update('x = x + 1'), frame)
    assert cf is not None
    assert cf.h0 == 9.0
    # And the closed form then steps from that binding.
    assert cf.kernel(3, cf.h0, cf.coeffs) == 12.0


# -- attention: nonlinear / self-nonlinear / unmodeled -> None ---------------


def test_nonlinear_self_reference_falls_to_attention():
    # x = x*x is not affine in x.
    assert AffineFlattener().match(_update('x = x*x'), _empty_frame()) is None


def test_nonlinear_call_on_self_falls_to_attention():
    # x = math.sin(x): the arg reads the target through an unmodeled call.
    assert AffineFlattener().match(_update('x = math.sin(x)'),
                                   _empty_frame()) is None


def test_uncompilable_coefficient_falls_to_attention():
    # b reads a live local the ConstSurface cannot resolve -> not compilable.
    assert AffineFlattener().match(_update('x = x + live_local'),
                                   _empty_frame()) is None


def test_division_by_self_falls_to_attention():
    assert AffineFlattener().match(_update('x = 1/x'), _empty_frame()) is None


def test_field_target_is_not_matched():
    # self.x = self.x + 1: a field target, not a bare frame variable this pass
    # models -> no affine match (attention).
    assert AffineFlattener().match(_update('self.x = self.x + 1'),
                                   _empty_frame()) is None


# -- route(): the registry -----------------------------------------------------


def test_route_returns_closed_form_when_a_flattener_matches():
    cf = route(_update('x = x*-1'), _empty_frame())
    assert cf is not None
    assert cf.kernel is affine_kernel


def test_route_returns_none_when_nothing_matches():
    assert route(_update('x = x*x'), _empty_frame()) is None


def test_route_default_registry_is_affine_only():
    assert len(flatteners.FLATTENERS) == 1
    assert isinstance(flatteners.FLATTENERS[0], AffineFlattener)


def test_route_honours_surface_argument():
    surface = ConstSurface({'k': 5})
    cf = route(_update('P1:rotationz(k)'), _empty_frame(), surface=surface)
    assert cf is not None
    assert cf.coeffs['b'].at(0.0) == 5.0


def test_route_registry_extends_without_core_change():
    # A new flattener class registers by prepending to the list route() is
    # given; first match wins, proving the registry is open for extension.
    calls = []

    class TaggingFlattener:
        def match(self, update, frame):
            calls.append(update.name)
            return ClosedForm(kernel=lambda n, h0, c: 'tagged', coeffs={}, h0=0)

    cf = route(_update('x = x + 1'), _empty_frame(),
               flatteners=[TaggingFlattener(), AffineFlattener()])
    assert cf is not None
    assert cf.kernel(0, 0, {}) == 'tagged'
    assert calls == ['x = x + 1']


def test_route_falls_through_to_next_flattener():
    class PassingFlattener:
        def match(self, update, frame):
            return None

    cf = route(_update('x = x + 1'), _empty_frame(),
               flatteners=[PassingFlattener(), AffineFlattener()])
    assert cf is not None
    assert cf.coeffs['a'].at(0.0) == 1.0
