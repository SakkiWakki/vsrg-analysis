"""Tree-walk evaluator - the reference/debug backend over the AST.

`tree_eval(node, surface)` recursively evaluates an expression against a
`Surface`, returning a concrete value or `UNRESOLVED`. It is the oracle the
compile backend is checked against and the readable path when a compiled
guard misbehaves; it re-queries the surface per node, so it is fine for
tests but too slow for a 60k-tick integration hot loop (that is what the
compile backend is for).

UNRESOLVED propagation:
- arithmetic / comparison: any UNRESOLVED operand -> UNRESOLVED.
- `not x`: UNRESOLVED -> UNRESOLVED.
- `and`: `False and _` -> False; `True and y` -> y; a resolved-true side
  with an UNRESOLVED other side -> UNRESOLVED (cannot prove live -> skip).
- `or`: `True or _` -> True; `False or y` -> y; UNRESOLVED unless the other
  side is resolved-true.
This is what lets window extraction drop a proven-true state conjunct
(`fgcurcommand == 2`) while keeping the `beat` range, and skip a guard whose
conjunct it cannot resolve.
"""
from __future__ import annotations

import operator

from analysis.player.render.expr import ast
from analysis.player.render.expr.surface import UNRESOLVED, Surface

_CMP = {
    '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge,
    '==': operator.eq, '~=': operator.ne,
}
_ARITH = {
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': operator.truediv, '%': operator.mod, '^': operator.pow,
}


def tree_eval(node: ast.Node, surface: Surface):
    match node:
        case ast.Num(value=v):
            return v
        case ast.Str(value=v):
            return v
        case ast.Bool(value=v):
            return v
        case ast.Nil():
            return None
        case ast.Sym(name=name):
            return surface.symbol(name)
        case ast.Index(base=base, key=key):
            b = tree_eval(base, surface)
            k = tree_eval(key, surface)
            if b is UNRESOLVED or k is UNRESOLVED:
                return UNRESOLVED
            return surface.index(b, k)
        case ast.Field(base=base, name=name):
            b = tree_eval(base, surface)
            return UNRESOLVED if b is UNRESOLVED else surface.index(b, name)
        case ast.Unary(op=op, operand=operand):
            return _eval_unary(op, tree_eval(operand, surface))
        case ast.Binary(op='and', left=l, right=r):
            return _eval_and(tree_eval(l, surface), lambda: tree_eval(r, surface))
        case ast.Binary(op='or', left=l, right=r):
            return _eval_or(tree_eval(l, surface), lambda: tree_eval(r, surface))
        case ast.Binary(op=op, left=l, right=r):
            return _eval_binary(op, tree_eval(l, surface), tree_eval(r, surface))
        case ast.Call(fn=ast.Sym(name=name), args=args):
            vals = [tree_eval(a, surface) for a in args]
            return UNRESOLVED if UNRESOLVED in vals else surface.call(name, vals)
        case _:
            return UNRESOLVED


def _eval_unary(op: str, x):
    if x is UNRESOLVED:
        return UNRESOLVED
    if op == '-':
        return -x
    if op == 'not':
        return not x
    if op == '#':
        try:
            return len(x)
        except TypeError:
            return UNRESOLVED
    return UNRESOLVED


def _eval_binary(op: str, a, b):
    if a is UNRESOLVED or b is UNRESOLVED:
        return UNRESOLVED
    fn = _CMP.get(op) or _ARITH.get(op)
    if fn is None:
        return UNRESOLVED
    try:
        return fn(a, b)
    except (TypeError, ZeroDivisionError):
        return UNRESOLVED


def _eval_and(a, rhs):
    if a is UNRESOLVED:
        # UNRESOLVED and y: if y is resolved-false the whole thing is false,
        # else we cannot prove it -> UNRESOLVED.
        b = rhs()
        return False if (b is not UNRESOLVED and not b) else UNRESOLVED
    if not a:
        return False
    return rhs()


def _eval_or(a, rhs):
    if a is UNRESOLVED:
        b = rhs()
        return True if (b is not UNRESOLVED and b) else UNRESOLVED
    if a:
        return True
    return rhs()
