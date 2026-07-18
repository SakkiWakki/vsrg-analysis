"""Recurrence flatteners: the extensible closed-form router core.

The NotITG Update body is compiled per frame variable. A per-frame update
`x = a*x + b` is a LINEAR RECURRENCE - the same object a compiler's induction
variable recognition turns into a CHAIN OF RECURRENCES (Bachmann/Zima 1994;
what LLVM's ScalarEvolution calls a SCEVAddRec). A first-order,
constant-coefficient chain of recurrence unrolls to a closed form evaluable at
any step n with no stepping. A `ClosedForm{kernel, coeffs, h0}` IS that
first-order constant-coefficient chain of recurrence.

The router is a REGISTRY of flatteners, tried in order, first match wins; when
none match, the update is left to be EVALUATED (interpreted) - the
always-correct floor, the analogue of ScalarEvolution returning SCEVUnknown for
a value it cannot characterize. Adding a recurrence class (modular counter,
phase oscillator) is appending a Flattener; the router core and the IR never
change.

`AffineFlattener` is the only class now. It matches `x = a*x + b` and its
degenerate forms (toggle `x*-1`, sum `x+step`, geometric `a*x+b`) plus the
pure-curve poke `p:prop(expr)` where `expr` does not read the target, which is
the a=0 case (value = b = the compiled curve). Coefficients `a` and `b` compile
to `scheduler.Channel`s over a `Surface`; an uncompilable coefficient (a live
local, an unmodeled call) means the update is not affine-flattenable and falls
to evaluation.

Stage A ASSUMPTION: a and b are treated as CONSTANT over the routed window -
the kernel evaluates each coefficient Channel once at the window start. A
time-varying coefficient is a Stage B widening (a variable-coefficient chain of
recurrence); the closed forms below are the constant-coefficient case.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from analysis.games.notitg.frame_ir import ClosedForm, Frame, VarUpdate
from analysis.player.render.expr import ast
from analysis.player.render.expr.compile_sched import compile_guard
from analysis.player.render.expr.surface import ConstSurface, Surface
from analysis.player.render.scheduler import Channel


@runtime_checkable
class Flattener(Protocol):
    """Recognizes one recurrence shape. `match` returns a `ClosedForm` (a
    chain-of-recurrence recipe: kernel + coefficient Channels + h0) when the
    update fits its shape, or None to pass to the next flattener."""

    def match(self, update: VarUpdate, frame: Frame) -> ClosedForm | None:
        ...


# -- closed-form kernel ------------------------------------------------------
#
# The recurrence is x_n = a*x_{n-1} + b, x_0 = h0. Its closed forms (standard
# chain-of-recurrence unrolling):
#   a == 1:  x_n = h0 + n*b
#   a != 1:  x_n = a^n * h0 + b*(a^n - 1)/(a - 1)
# a == -1 is NOT special-cased: it is the a != 1 formula with a = -1, which
# gives the even-n -> h0, odd-n -> -h0 + b behaviour for free. Routing it
# through the general branch keeps the b-term in one place (fewer ways to get
# it wrong). a == 0 (pure curve) is also the a != 1 branch: x_n = b for n >= 1;
# a pure-curve poke sets h0 = b(0) so x_0 = b too.
#
# The kernel is `(n, h0, coeffs) -> value`; `coeffs` is {'a': Channel, 'b':
# Channel}. Per the Stage A assumption a and b are sampled once at the window
# start (t = 0.0 in the coefficient Channel's own coordinate) - constant over
# the window.


def _coeff(coeffs: dict, name: str, default: float = 0.0) -> float:
    channel = coeffs.get(name)
    if channel is None:
        return default
    return channel.at(0.0)


def affine_kernel(n: int, h0, coeffs: dict):
    """Closed form of x_n = a*x_{n-1} + b from x_0 = h0, a and b constant."""
    a = _coeff(coeffs, 'a', 0.0)
    b = _coeff(coeffs, 'b', 0.0)
    if a == 1:
        return h0 + n * b
    a_pow_n = a ** n
    return a_pow_n * h0 + b * (a_pow_n - 1) / (a - 1)


# -- affine decomposition ----------------------------------------------------


def _self_reads(node: ast.Node, target: str) -> bool:
    """True when `node`'s subtree reads `target` as a bare `Sym` (the RHS
    self-reference that makes an update a recurrence rather than a pure curve).
    An actor-prop target ('P1.rotationz') never appears as a readable Sym, so a
    poke arg self-reads only through a bare frame variable of that name."""
    match node:
        case ast.Sym(name=name):
            return name == target
        case ast.Unary(operand=operand):
            return _self_reads(operand, target)
        case ast.Binary(left=left, right=right):
            return _self_reads(left, target) or _self_reads(right, target)
        case ast.Call(fn=fn, args=args):
            return _self_reads(fn, target) or any(
                _self_reads(arg, target) for arg in args)
        case ast.Method(recv=recv, args=args):
            return _self_reads(recv, target) or any(
                _self_reads(arg, target) for arg in args)
        case ast.Index(base=base, key=key):
            return _self_reads(base, target) or _self_reads(key, target)
        case ast.Field(base=base):
            return _self_reads(base, target)
    return False


def _target_sym(node: ast.Node) -> str | None:
    """The bare `Sym` name assigned by an `Assign`, or None (a field/index
    target is not a self-referential frame variable this pass models)."""
    match node:
        case ast.Assign(targets=(ast.Sym(name=name),)):
            return name
    return None


def _value_expr(node: ast.Node) -> ast.Node | None:
    """The single value expression written by the update: an `Assign`'s sole
    RHS, or a poke `recv:prop(expr)`'s sole argument."""
    match node:
        case ast.Assign(values=(value,)):
            return value
        case ast.ExprStmt(expr=ast.Method(args=(arg,))):
            return arg
    return None


_UNMODELED = ast.Unparsed('unmodeled-self-reference')


def _is_bare_target(node: ast.Node, target: str) -> bool:
    return isinstance(node, ast.Sym) and node.name == target


def _split_affine(value: ast.Node,
                  target: str) -> tuple[ast.Node | None, ast.Node]:
    """Decompose `value` into (a_node, b_node) so value == a*target + b, with
    `a_node` None meaning a == 0 (target does not appear). Recognizes:
      target                  -> a=1,  b=0
      -target                 -> a=-1, b=0
      k * target / target * k -> a=k,  b=0
      target + t / t + target -> a=1,  b=t   (recurse on the non-target side)
      target - t              -> a=1,  b=-t
      t - target              -> a=-1, b=t
    An expression that reads `target` in a shape this does not model returns
    (_UNMODELED, value) so the caller routes to evaluation."""
    if not _self_reads(value, target):
        return None, value
    match value:
        case ast.Sym(name=name) if name == target:
            return ast.Num(1.0), ast.Num(0.0)
        case ast.Unary(op='-', operand=operand) if _is_bare_target(operand, target):
            return ast.Num(-1.0), ast.Num(0.0)
        case ast.Binary(op='*', left=left, right=right):
            return _split_mul(left, right, target)
        case ast.Binary(op='+', left=left, right=right):
            return _split_add(left, right, target)
        case ast.Binary(op='-', left=left, right=right):
            return _split_sub(left, right, target)
    return _UNMODELED, value


def _split_mul(left: ast.Node, right: ast.Node,
               target: str) -> tuple[ast.Node | None, ast.Node]:
    left_self, right_self = _self_reads(left, target), _self_reads(right, target)
    if left_self and not right_self and _is_bare_target(left, target):
        return right, ast.Num(0.0)
    if right_self and not left_self and _is_bare_target(right, target):
        return left, ast.Num(0.0)
    return _UNMODELED, ast.Num(0.0)


def _split_add(left: ast.Node, right: ast.Node,
               target: str) -> tuple[ast.Node | None, ast.Node]:
    left_self, right_self = _self_reads(left, target), _self_reads(right, target)
    if left_self and not right_self:
        a_node, b_node = _split_affine(left, target)
        return a_node, _sum(b_node, right)
    if right_self and not left_self:
        a_node, b_node = _split_affine(right, target)
        return a_node, _sum(b_node, left)
    return _UNMODELED, ast.Num(0.0)


def _split_sub(left: ast.Node, right: ast.Node,
               target: str) -> tuple[ast.Node | None, ast.Node]:
    left_self, right_self = _self_reads(left, target), _self_reads(right, target)
    if left_self and not right_self:
        a_node, b_node = _split_affine(left, target)
        return a_node, _sum(b_node, ast.Unary('-', right))
    if right_self and not left_self:
        a_node, b_node = _split_affine(right, target)
        return _negate(a_node), left
    return _UNMODELED, ast.Num(0.0)


def _sum(left: ast.Node, right: ast.Node) -> ast.Node:
    if isinstance(left, ast.Num) and left.value == 0.0:
        return right
    return ast.Binary('+', left, right)


def _negate(node: ast.Node | None) -> ast.Node:
    if node is None:
        return ast.Num(0.0)
    if isinstance(node, ast.Num):
        return ast.Num(-node.value)
    return ast.Unary('-', node)


# -- the affine flattener ----------------------------------------------------


class AffineFlattener:
    """Flattens `x = a*x + b` and its degenerate forms to a constant-coefficient
    chain of recurrence. The pure-curve poke (no self-reference) is the a=0
    case: value = b = the compiled curve."""

    def __init__(self, surface: Surface | None = None):
        self._surface = surface if surface is not None else ConstSurface()

    def match(self, update: VarUpdate, frame: Frame) -> ClosedForm | None:
        value = _value_expr(update.node)
        if value is None:
            return None
        target = _target_sym(update.node)
        if target is None:
            a_node, b_node = None, value
        else:
            a_node, b_node = _split_affine(value, target)
            if a_node is _UNMODELED:
                return None
        return self._closed_form(a_node, b_node, target, frame)

    def _closed_form(self, a_node: ast.Node | None, b_node: ast.Node,
                     target: str | None, frame: Frame) -> ClosedForm | None:
        a_channel = (self._channel(a_node) if a_node is not None
                     else _const_channel(0.0))
        b_channel = self._channel(b_node)
        if a_channel is None or b_channel is None:
            return None
        h0 = self._initial(target, frame, b_channel)
        return ClosedForm(kernel=affine_kernel,
                          coeffs={'a': a_channel, 'b': b_channel}, h0=h0)

    def _channel(self, node: ast.Node) -> Channel | None:
        return compile_guard(node, self._surface)

    def _initial(self, target: str | None, frame: Frame, b_channel: Channel):
        """h0 is the value the recurrence starts from: a frame binding for the
        target if one is in scope, else the pure-curve poke's own value b(0)."""
        if target is not None:
            bound = _resolve(target, frame)
            if bound is not None:
                return bound
        return b_channel.at(0.0)


# -- name resolution up the frame chain --------------------------------------


def _resolve(name: str, frame: Frame | None):
    """Walk parent frames for `name`'s binding (the interpreter's chain lookup),
    or None when unbound."""
    while frame is not None:
        if name in frame.bindings:
            return frame.bindings[name]
        frame = frame.parent
    return None


def _const_channel(value: float) -> Channel:
    return Channel(curve=lambda coord: value)


# -- the router --------------------------------------------------------------

FLATTENERS: list[Flattener] = [AffineFlattener()]


def route(update: VarUpdate, frame: Frame, surface: Surface | None = None,
          flatteners: list[Flattener] | None = None) -> ClosedForm | None:
    """Try each registered flattener in order; the first `ClosedForm` wins and
    routes the update to a closed form. None means no flattener matched -> evaluation (the
    always-correct floor). `surface` re-binds the default registry to a live
    surface; pass `flatteners` to override the registry entirely."""
    if flatteners is None:
        flatteners = ([AffineFlattener(surface)] if surface is not None
                      else FLATTENERS)
    for flattener in flatteners:
        closed_form = flattener.match(update, frame)
        if closed_form is not None:
            return closed_form
    return None
