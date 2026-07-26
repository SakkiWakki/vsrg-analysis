"""AST -> flat op-stream compiler (the upstream SANITIZER for the C executor).

This lowers a parsed Update body (`expr.parser.parse_body` -> stmts) into a flat,
validated op array the C computed-goto executor runs with NO per-node
interpretation: variables are resolved to frame SLOTS (not names, killing the
scope-chain walk), operators to op-ids (not re-dispatch), control flow to JUMPS.

Semantics oracle: `expr.frame_compile_exec` (op-for-op) + `expr.frame_eval`. Any
node shape outside the modeled subset lowers to a FALLBACK op that the executor
routes back to the Python interpreter - the strict never-a-coverage-regression
floor (covers FuncExpr/FuncDef closures = 1% of charts, Return-with-value, etc.).

The op array is a list of fixed-shape records; `serialize()` packs them into the
buffers the C side consumes. Emitting the array is one-time-per-chart (not hot),
so it stays readable Python; C owns the per-tick execution.
"""
from __future__ import annotations

from analysis.player.render.expr import ast


# --- op codes (must match exec.c's DISPATCH order) -----------------------
# Each op is (OPCODE, a, b, c) - up to three integer operands (slot ids, const
# ids, jump targets, op-ids, arg counts). String/number literals live in pools.
class Op:
    # expressions push one value onto the register stack
    CONST        = 0    # a=const-pool id
    LOAD_SLOT    = 1    # a=slot index
    STORE_SLOT   = 2    # a=slot index (pops)
    LOAD_GLOBAL  = 3    # a=global-name id (accumulator store)
    STORE_GLOBAL = 4    # a=global-name id (pops)
    LOAD_SYMBOL  = 5    # a=name id -> frontier.symbol (beat/actor-global/host fn)
    BINARY       = 6    # a=binary op-id (cvalue_ops COP_*)
    UNARY        = 7    # a=unary op-id (CUN_*)
    INDEX        = 8    # pops base,key -> value
    FIELD        = 9    # a=name id; pops base -> value
    GETTER       = 10   # a=verb name id, b=argc; pops recv,args -> value (frontier)
    METHOD       = 11   # a=verb name id, b=argc; generic method (GetChild/GetShader)
    CALL_SYM     = 12   # a=name id, b=argc; a free call name(args)
    CALL_MATH    = 13   # a=math fn id, b=argc; native math.<fn>
    CALL_BUILTIN = 14   # a=builtin id, b=argc; type/tonumber/tostring/table.*
    MAKE_TABLE   = 15   # a=n-array, b=n-fields (field keys in const pool run)
    NEWTABLE_ARR = 16   # a=prealloc size (an empty {} or {a,b,c})
    POKE         = 17   # a=verb id, b=argc; pops recv,args (effect, no push)
    SET_INDEX    = 18   # pops base,key,value (t[k]=v)
    SET_FIELD    = 19   # a=name id; pops base,value (t.f=v)
    POP          = 20   # discard top (ExprStmt value)
    JUMP         = 21   # a=target
    JUMP_IF_FALSE= 22   # a=target; pops cond (control truthiness)
    JIF_FALSE_KEEP = 23 # a=target; PEEKS cond (and/or: leaves operand if jumping)
    JIF_TRUE_KEEP  = 24 # a=target; PEEKS cond
    DUP          = 25   # duplicate top (and/or operand plumbing)
    RETURN_HALT  = 26   # end the body (valueless top-level return)
    ITER_SETUP   = 27   # a=n-exprs; set up a generic-for iterator on the stack
    ITER_NEXT    = 28   # a=first loop-var slot, b=n-vars, c=end-target
    FALLBACK     = 29   # a=node-pool id; route this AST node to the interpreter
    TABLE_INSERT = 32   # pops table,value; append (arena) or frontier host table
    CALL_VALUE   = 33   # a=argc; pops fn,args -> call a computed callable
    GET_PROP     = 34   # a=prop id; pops recv -> a live actor property read
    SET_PROP     = 35   # a=setter id; pops recv,value -> a live actor write


# native math fns the analytic/native path supports (compile-resolved).
_MATH_FNS = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh',
             'cosh', 'tanh', 'exp', 'log', 'log10', 'sqrt', 'abs', 'floor',
             'ceil', 'fmod', 'pow', 'deg', 'rad', 'min', 'max', 'random']
_MATH_ID = {n: i for i, n in enumerate(_MATH_FNS)}
_MATH_CONST = {'pi': 3.141592653589793, 'huge': float('inf')}

_BINOP = {'+': 0, '-': 1, '*': 2, '/': 3, '%': 4, '^': 5, '<': 6, '<=': 7,
          '>': 8, '>=': 9, '==': 10, '~=': 11, '..': 12}
_UNOP = {'-': 0, 'not': 1, '#': 2}

_BUILTINS = ['type', 'tonumber', 'tostring']  # table.* handled via Field+Call
_BUILTIN_ID = {n: i for i, n in enumerate(_BUILTINS)}


class _Scope:
    """Compile-time lexical scope: name -> slot index. Parent chain mirrors the
    runtime Scope so a Sym resolves to the nearest declaring slot."""

    def __init__(self, parent=None):
        self.names: dict[str, int] = {}
        self.parent = parent

    def resolve(self, name):
        s = self
        while s is not None:
            if name in s.names:
                return s.names[name]
            s = s.parent
        return None


class Compiler:
    """One body -> one OpProgram. `writes` (names assigned but never locally
    declared) become the accumulator globals; every other free Sym is a frontier
    symbol."""

    def __init__(self):
        self.ops: list[tuple] = []
        self.consts: list = []          # const pool (numbers/strings/bool/nil)
        self.names: list[str] = []      # interned names (verbs, globals, syms)
        self.nodes: list = []           # FALLBACK node pool
        self._name_id: dict[str, int] = {}
        self.nslots = 0
        # Names read as bare frontier symbols (LOAD_SYMBOL) and never written -
        # candidates to snapshot into the arena so nested v[i][j] indexing never
        # crosses back to Python. `writes` are excluded when compile() runs.
        self._symbol_reads: set[str] = set()
        # Inlining state: the helpers available, the ones currently being
        # inlined (a self-call falls back to a crossing rather than recursing
        # forever), and a stack of [result_slot, exit_jump_sites] frames.
        self._inline_fns: dict = {}
        self._prop_gets: dict = {}
        self._prop_sets: dict = {}
        self._prop_verbs: tuple = ()
        self._inlining: set = set()
        self._inline_frames: list = []

    # -- pools --
    def _const(self, value):
        self.consts.append(value)
        return len(self.consts) - 1

    def _name(self, s):
        i = self._name_id.get(s)
        if i is None:
            i = len(self.names)
            self.names.append(s)
            self._name_id[s] = i
        return i

    def _node(self, n):
        self.nodes.append(n)
        return len(self.nodes) - 1

    def _emit(self, opcode, a=0, b=0, c=0):
        self.ops.append((opcode, a, b, c))
        return len(self.ops) - 1

    def _patch(self, at, a=None, b=None, c=None):
        op = self.ops[at]
        self.ops[at] = (op[0],
                        op[1] if a is None else a,
                        op[2] if b is None else b,
                        op[3] if c is None else c)

    def _slot(self, scope, name):
        """Declare `name` a local in `scope`, returning its slot."""
        if name in scope.names:
            return scope.names[name]
        idx = self.nslots
        self.nslots += 1
        scope.names[name] = idx
        return idx

    # ----------------------------------------------------------------
    def compile(self, stmts, global_writes: set, receivers=None,
                inline_fns=None, prop_gets=None, prop_sets=None):
        self._global_writes = global_writes
        self._receivers = receivers
        self._inline_fns = inline_fns or {}
        self._prop_gets = prop_gets or {}
        self._prop_sets = prop_sets or {}
        # id -> verb, the inverse the frontier needs for a non-actor receiver.
        self._prop_verbs = tuple(
            v for v, _i in sorted(self._prop_gets.items(), key=lambda kv: kv[1]))
        self._set_verbs = tuple(
            v for v, _i in sorted(self._prop_sets.items(), key=lambda kv: kv[1]))
        root = _Scope()
        # slot 0 is reserved for `self` (rebound each tick)
        self._slot(root, 'self')
        for s in stmts:
            self._stmt(s, root)
        self._emit(Op.RETURN_HALT)
        return OpProgram(self.ops, self.consts, self.names, self.nodes,
                         self.nslots, self._symbol_reads, self._receivers,
                         self._prop_verbs, self._set_verbs)

    # -- statements --
    def _stmt(self, node, scope):
        t = type(node)
        if t is ast.Local:
            self._local(node, scope)
        elif t is ast.Assign:
            self._assign(node, scope)
        elif t is ast.ExprStmt:
            # A Method in STATEMENT position is a POKE (effect), not a getter
            # read - matches frame_compile_exec._compile_expr_stmt. `self:zoom(x)`
            # / `P1:rotationz(..)` mutate the actor; only a Method used as a VALUE
            # is a getter. Everything else: evaluate + discard.
            e = node.expr
            if type(e) is ast.Method:
                # A setter the host published as a plain property write lowers
                # to SET_PROP with an integer id - the verb is a compile-time
                # constant, so resolving it per tick is work that need not
                # happen. One argument only: the multi-property verbs (zoom,
                # align) write two properties from one call and keep crossing.
                set_id = self._prop_sets.get(e.name)
                if set_id is not None and len(e.args) == 1:
                    self._expr(e.recv, scope)
                    self._expr(e.args[0], scope)
                    self._emit(Op.SET_PROP, set_id)
                else:
                    self._expr(e.recv, scope)
                    for arg in e.args:
                        self._expr(arg, scope)
                    self._emit(Op.POKE, self._name(e.name), len(e.args))
            else:
                self._expr(e, scope, want_value=True)
                self._emit(Op.POP)
        elif t is ast.If:
            self._if(node, scope)
        elif t is ast.NumericFor:
            self._numeric_for(node, scope)
        elif t is ast.While:
            self._while(node, scope)
        elif t is ast.GenericFor:
            self._generic_for(node, scope)
        elif t is ast.Return and self._inline_frames:
            self._inline_return(node, scope)
        elif t is ast.Return and not node.values:
            self._emit(Op.RETURN_HALT)
        else:
            # FuncDef, Return-with-value, Unparsed, anything unmodeled ->
            # interpreter fallback (never a coverage regression).
            self._emit(Op.FALLBACK, self._node(node))

    def _inline_return(self, node, scope):
        """`return` inside an INLINED helper stores its value and jumps to the
        inline's end. It must NOT emit RETURN_HALT - that ends the whole tick,
        not the helper."""
        frame = self._inline_frames[-1]
        if node.values:
            self._expr(node.values[0], scope)
        else:
            self._emit(Op.CONST, self._const(None))
        self._emit(Op.STORE_SLOT, frame[0])
        frame[1].append(self._emit(Op.JUMP))

    def _inline_call(self, node, name, scope) -> bool:
        """Compile `name(args)` as the helper's BODY in place of a crossing.

        The helper's scope is a FRESH root, never a child of the call site: a
        top-level Lua function sees globals and its own params, and must not
        capture the caller's locals. Arguments evaluate left to right in the
        CALLER's scope, then bind - extras are evaluated and discarded, missing
        ones bind nil, both as Lua does."""
        fn = self._inline_fns.get(name)
        if fn is None or name in self._inlining \
                or len(self._inline_frames) >= _INLINE_DEPTH:
            return False

        inner = _Scope()
        result = self.nslots
        self.nslots += 1
        slots = [self._slot(inner, p) for p in fn.params]

        for arg in node.args:
            self._expr(arg, scope)
        for _extra in range(len(node.args) - len(slots)):
            self._emit(Op.POP)
        for slot in reversed(slots[:len(node.args)]):
            self._emit(Op.STORE_SLOT, slot)
        for slot in slots[len(node.args):]:
            self._emit(Op.CONST, self._const(None))
            self._emit(Op.STORE_SLOT, slot)

        # A body that falls off the end yields nil.
        self._emit(Op.CONST, self._const(None))
        self._emit(Op.STORE_SLOT, result)

        frame = [result, []]
        self._inline_frames.append(frame)
        self._inlining.add(name)
        for stmt in fn.body:
            self._stmt(stmt, inner)
        self._inlining.discard(name)
        self._inline_frames.pop()

        for site in frame[1]:
            self._patch(site, a=len(self.ops))
        self._emit(Op.LOAD_SLOT, result)
        return True

    def _local(self, node, scope):
        # evaluate RHS values, then bind slots (Lua: RHS sees the OLD bindings)
        for i, name in enumerate(node.names):
            if i < len(node.values):
                self._expr(node.values[i], scope, want_value=True)
            else:
                self._emit(Op.CONST, self._const(None))
            slot = self._slot(scope, name)
            self._emit(Op.STORE_SLOT, slot)

    def _assign(self, node, scope):
        # eval all RHS (left-to-right), then assign each target
        n = len(node.targets)
        for i in range(n):
            if i < len(node.values):
                self._expr(node.values[i], scope, want_value=True)
            else:
                self._emit(Op.CONST, self._const(None))
        # values now on stack in order; assign in reverse so top matches last
        for i in reversed(range(n)):
            self._assign_target(node.targets[i], scope)

    def _assign_target(self, target, scope):
        t = type(target)
        if t is ast.Sym:
            slot = scope.resolve(target.name)
            if slot is not None:
                self._emit(Op.STORE_SLOT, slot)
            else:
                # implicit global (Lua rule) -> accumulator store
                self._emit(Op.STORE_GLOBAL, self._name(target.name))
        elif t is ast.Index:
            # t[k] = v : need base,key on stack UNDER the value. Re-eval base+key.
            # stack currently: [.., value]. Push base,key then SET_INDEX which
            # pops value,key,base in its own order.
            self._expr(target.base, scope, want_value=True)
            self._expr(target.key, scope, want_value=True)
            self._emit(Op.SET_INDEX)
        elif t is ast.Field:
            self._expr(target.base, scope, want_value=True)
            self._emit(Op.SET_FIELD, self._name(target.name))
        else:
            self._emit(Op.FALLBACK, self._node(target))

    def _if(self, node, scope):
        end_jumps = []
        # cond
        self._expr(node.cond, scope, want_value=True)
        jf = self._emit(Op.JUMP_IF_FALSE)
        self._block(node.body, scope)
        end_jumps.append(self._emit(Op.JUMP))
        self._patch(jf, a=len(self.ops))
        for econd, ebody in node.elifs:
            self._expr(econd, scope, want_value=True)
            jf = self._emit(Op.JUMP_IF_FALSE)
            self._block(ebody, scope)
            end_jumps.append(self._emit(Op.JUMP))
            self._patch(jf, a=len(self.ops))
        if node.orelse:
            self._block(node.orelse, scope)
        for j in end_jumps:
            self._patch(j, a=len(self.ops))

    def _numeric_for(self, node, scope):
        # slots for i, limit, step in the LOOP scope
        loop = _Scope(scope)
        islot = self._slot(loop, node.var)
        # lim and stp are allocated CONSECUTIVELY (stp == lim+1). NUMFOR_TEST's
        # record has only a/b/c (i-slot, lim-slot, exit-target after patch), so
        # the executor reads the step slot as lim+1 - this consecutive-alloc
        # contract is the shared invariant between here and exec.c.
        lim = self.nslots; self.nslots += 1
        stp = self.nslots; self.nslots += 1
        assert stp == lim + 1
        self._expr(node.start, scope, want_value=True)
        self._emit(Op.STORE_SLOT, islot)
        self._expr(node.stop, scope, want_value=True)
        self._emit(Op.STORE_SLOT, lim)
        if node.step is not None:
            self._expr(node.step, scope, want_value=True)
        else:
            self._emit(Op.CONST, self._const(1.0))
        self._emit(Op.STORE_SLOT, stp)
        # NUMFOR_TEST is a fused, sign-correct loop guard (i<=lim if step>0 else
        # i>=lim) run in C - keeps the loop interpretation-free. c is patched to
        # the loop-exit target. NUMFOR_STEP does i += step. No stack use (all
        # slot-addressed), so the loop is stack-balanced.
        header = len(self.ops)
        test = self._emit(_NUMFOR_TEST, islot, lim, stp)
        self._block(node.body, loop)
        self._emit(_NUMFOR_STEP, islot, stp)
        self._emit(Op.JUMP, header)
        self._patch(test, c=len(self.ops))

    def _while(self, node, scope):
        header = len(self.ops)
        self._expr(node.cond, scope, want_value=True)
        jf = self._emit(Op.JUMP_IF_FALSE)
        self._block(node.body, _Scope(scope))
        self._emit(Op.JUMP, header)
        self._patch(jf, a=len(self.ops))

    def _generic_for(self, node, scope):
        # for names in ipairs(t)/pairs(t) do ... end - the ONLY generic-for form
        # the corpus uses (matches frame_eval._iter_pairs). Detect ipairs/pairs,
        # evaluate the TABLE arg (not the call), push a mode flag, and let
        # ITER_SETUP build the iterator over that table (arena native or host via
        # the frontier). A generic-for over anything else -> FALLBACK.
        loop = _Scope(scope)
        slots = [self._slot(loop, nm) for nm in node.names]
        e0 = node.exprs[0] if node.exprs else None
        mode = None
        if type(e0) is ast.Call and type(e0.fn) is ast.Sym \
                and e0.fn.name in ('ipairs', 'pairs') and e0.args:
            mode = 0 if e0.fn.name == 'ipairs' else 1
            self._expr(e0.args[0], scope)        # the table
            self._emit(Op.ITER_SETUP, mode, 1)   # b=1 -> ipairs/pairs form
        else:
            self._emit(Op.FALLBACK, self._node(node))
            return
        top = len(self.ops)
        nx = self._emit(Op.ITER_NEXT, slots[0] if slots else 0, len(slots))
        self._block(node.body, loop)
        self._emit(Op.JUMP, top)
        self._patch(nx, c=len(self.ops))

    def _block(self, stmts, scope):
        for s in stmts:
            self._stmt(s, scope)

    # -- expressions (each leaves exactly one value on the register stack) --
    def _expr(self, node, scope, want_value=True):
        t = type(node)
        if t is ast.Num:
            self._emit(Op.CONST, self._const(float(node.value)))
        elif t is ast.Str:
            self._emit(Op.CONST, self._const(node.value))
        elif t is ast.Bool:
            self._emit(Op.CONST, self._const(bool(node.value)))
        elif t is ast.Nil:
            self._emit(Op.CONST, self._const(None))
        elif t is ast.Sym:
            self._sym(node, scope)
        elif t is ast.Index:
            self._expr(node.base, scope); self._expr(node.key, scope)
            self._emit(Op.INDEX)
        elif t is ast.Field:
            self._field(node, scope)
        elif t is ast.Binary:
            self._binary(node, scope)
        elif t is ast.Unary:
            self._expr(node.operand, scope)
            self._emit(Op.UNARY, _UNOP.get(node.op, 0))
        elif t is ast.Method:
            self._method(node, scope)
        elif t is ast.Call:
            self._call(node, scope)
        elif t is ast.Table:
            self._table(node, scope)
        else:
            # FuncExpr and anything unmodeled -> fallback (pushes its value)
            self._emit(Op.FALLBACK, self._node(node))

    def _sym(self, node, scope):
        slot = scope.resolve(node.name)
        if slot is not None:
            self._emit(Op.LOAD_SLOT, slot)
        elif node.name in self._global_writes:
            self._emit(Op.LOAD_GLOBAL, self._name(node.name))
        else:
            # a frontier symbol: driver clock (beat), an actor global, a host fn
            self._symbol_reads.add(node.name)
            self._emit(Op.LOAD_SYMBOL, self._name(node.name))

    def _field(self, node, scope):
        # math.pi / math.huge fold to a constant
        if type(node.base) is ast.Sym and node.base.name == 'math' \
                and node.name in _MATH_CONST:
            self._emit(Op.CONST, self._const(_MATH_CONST[node.name]))
            return
        self._expr(node.base, scope)
        self._emit(Op.FIELD, self._name(node.name))

    def _binary(self, node, scope):
        if node.op in ('and', 'or'):
            self._and_or(node, scope)
            return
        self._expr(node.left, scope)
        self._expr(node.right, scope)
        self._emit(Op.BINARY, _BINOP.get(node.op, 12))

    def _and_or(self, node, scope):
        # Lua returns the OPERAND. `a and b`: if a falsy -> a, else b. UNRESOLVED
        # a -> UNRESOLVED (poison handled in C JIF_*_KEEP). Short-circuit:
        self._expr(node.left, scope)
        if node.op == 'and':
            jmp = self._emit(Op.JIF_FALSE_KEEP)   # if a falsy, keep a, skip b
        else:
            jmp = self._emit(Op.JIF_TRUE_KEEP)    # if a truthy, keep a, skip b
        self._emit(Op.POP)                        # discard a, evaluate b
        self._expr(node.right, scope)
        self._patch(jmp, a=len(self.ops))

    def _method(self, node, scope):
        # recv:verb(args). A getter (returns a value) vs a poke (effect) is
        # decided by whether the result is used - here in expr position it is a
        # GETTER read. Push recv, args, then GETTER.
        #
        # A verb the host published as a plain PROPERTY read lowers to GET_PROP
        # carrying an integer id instead. The verb is a compile-time constant,
        # so resolving it to a property has no business happening per tick: the
        # generic GETTER crosses with a string the host must decode and then
        # route through its verb tables, several thousand times a second, to
        # reach an answer fixed at compile time.
        prop_id = self._prop_gets.get(node.name)
        if prop_id is not None and not node.args:
            self._expr(node.recv, scope)
            self._emit(Op.GET_PROP, prop_id)
            return
        self._expr(node.recv, scope)
        for arg in node.args:
            self._expr(arg, scope)
        self._emit(Op.GETTER, self._name(node.name), len(node.args))

    def _call(self, node, scope):
        fn = node.fn
        # math.<fn>(...)
        if type(fn) is ast.Field and type(fn.base) is ast.Sym \
                and fn.base.name == 'math' and fn.name in _MATH_ID:
            for arg in node.args:
                self._expr(arg, scope)
            self._emit(Op.CALL_MATH, _MATH_ID[fn.name], len(node.args))
            return
        # a builtin free call type/tonumber/tostring
        if type(fn) is ast.Sym and fn.name in _BUILTIN_ID \
                and scope.resolve(fn.name) is None:
            for arg in node.args:
                self._expr(arg, scope)
            self._emit(Op.CALL_BUILTIN, _BUILTIN_ID[fn.name], len(node.args))
            return
        # table.insert/getn/remove : Field(base=Sym('table')). Lower natively -
        # these map to arena ops (getn == #t, insert == append). They run every
        # tick (table.getn in loop guards), so a FALLBACK here is both a coverage
        # gap AND a per-tick interp+exception cost.
        if type(fn) is ast.Field and type(fn.base) is ast.Sym \
                and fn.base.name == 'table':
            if fn.name == 'getn' and len(node.args) == 1:
                self._expr(node.args[0], scope)
                self._emit(Op.UNARY, _UNOP['#'])   # #t
                return
            if fn.name == 'insert' and len(node.args) == 2:
                self._expr(node.args[0], scope)     # table
                self._expr(node.args[1], scope)     # value
                self._emit(Op.TABLE_INSERT)
                # table.insert is a statement (no value); but a Call is an expr,
                # so push nil to keep the stack balanced (ExprStmt POPs it).
                self._emit(Op.CONST, self._const(None))
                return
            # table.remove / other -> fallback (rare)
            self._emit(Op.FALLBACK, self._node(node))
            return
        # a plain free call name(args) -> frontier / closure
        if type(fn) is ast.Sym and scope.resolve(fn.name) is None:
            if self._inline_call(node, fn.name, scope):
                return
            for arg in node.args:
                self._expr(arg, scope)
            self._emit(Op.CALL_SYM, self._name(fn.name), len(node.args))
            return
        # A COMPUTED callee: `a[3](beat)` (Index/Field fn), or a local slot
        # holding a closure. Evaluate the fn VALUE, push args, CALL_VALUE invokes
        # whatever it resolves to (a host closure crosses as a handle). This is
        # the mod-perframe painter idiom (mod_perframes[i][3](beat)).
        self._expr(fn, scope)
        for arg in node.args:
            self._expr(arg, scope)
        self._emit(Op.CALL_VALUE, len(node.args))

    def _table(self, node, scope):
        if node.fields:
            # keyed table constructor -> fallback (corpus: 0 hash constructors)
            self._emit(Op.FALLBACK, self._node(node))
            return
        # array constructor {a, b, c}
        for el in node.array:
            self._expr(el, scope)
        self._emit(Op.MAKE_TABLE, len(node.array), 0)


# numeric-for uses two dedicated fused opcodes (sign-correct test + step) so the
# loop stays interpretation-free; kept out of the public Op enum block for
# clarity but part of the same code space.
_NUMFOR_TEST = 30   # a=i-slot, b=lim-slot, extra=step-slot(in the record's c-slot pre-patch); c=end target after patch
_NUMFOR_STEP = 31   # a=i-slot, b=step-slot


class OpProgram:
    """The compiled body: op records + pools + slot count. `serialize()` packs
    into the flat buffers the C executor consumes."""

    def __init__(self, ops, consts, names, nodes, nslots, symbol_reads=None,
                 receivers=None, prop_verbs=(), set_verbs=()):
        self.ops = ops
        self.consts = consts
        self.names = names
        self.nodes = nodes
        self.nslots = nslots
        # Bare-symbol reads never written by the body: arena-snapshot candidates.
        self.symbol_reads = symbol_reads or set()
        # Which crossing-out sources can reach a method receiver - see
        # ReceiverSources. None when the caller did not ask for the analysis.
        self.receivers = receivers or ReceiverSources()
        # GET_PROP id -> source verb, so a non-actor receiver can still be
        # answered through the generic path.
        self.prop_verbs = tuple(prop_verbs)
        self.set_verbs = tuple(set_verbs)

    def summary(self):
        from collections import Counter
        c = Counter(op[0] for op in self.ops)
        return {'ops': len(self.ops), 'slots': self.nslots,
                'consts': len(self.consts), 'names': len(self.names),
                'fallbacks': c.get(Op.FALLBACK, 0), 'by_op': dict(c)}

    def serialize(self):
        """Pack into the flat buffers the C executor consumes:
          - ops:   int32[4*nops]  (opcode, a, b, c) per op
          - consts: a parallel pair (kind[i], numval[i], strval[i]) - kind 0 num,
                    1 str, 2 true, 3 false, 4 nil; the executor pre-materializes
                    each const into a CValue at load (strings interned into the
                    arena) so CONST is a single array load.
          - names:  the interned name strings (verbs/globals/symbols), joined
                    NUL-separated with an offsets array.
        Returns a plain dict of Python buffers; the ctypes layer marshals them.
        The FALLBACK node pool is NOT serialized to C - fallback ops carry the
        node id and the executor calls back into Python with it."""
        import struct
        # ops -> int32 quads
        ops_buf = bytearray()
        for (op, a, b, cc) in self.ops:
            ops_buf += struct.pack('<iiii', op, a, b, cc)
        # consts -> kinds + nums + a string table (only str consts populate it)
        ckinds = bytearray()
        cnums = bytearray()
        cstrs = []          # (const-index -> string) for str consts
        for v in self.consts:
            if isinstance(v, bool):
                ckinds.append(2 if v else 3)
                cnums += struct.pack('<d', 0.0)
            elif isinstance(v, (int, float)):
                ckinds.append(0)
                cnums += struct.pack('<d', float(v))
            elif isinstance(v, str):
                ckinds.append(1)
                cnums += struct.pack('<d', float(len(cstrs)))  # index into cstrs
                cstrs.append(v)
            elif v is None:
                ckinds.append(4)
                cnums += struct.pack('<d', 0.0)
            else:
                ckinds.append(4)
                cnums += struct.pack('<d', 0.0)
        return {
            'nops': len(self.ops),
            'ops': bytes(ops_buf),
            'nconsts': len(self.consts),
            'const_kinds': bytes(ckinds),
            'const_nums': bytes(cnums),
            'const_strs': cstrs,           # list[str] for str consts, in order
            'names': list(self.names),     # list[str], id == index
            'nslots': self.nslots,
        }


def _free_names(node, bound: set) -> set:
    """Names `node` reads that `bound` does not supply - its free variables."""
    free: set = set()
    scoped = set(bound)

    def scan(n):
        t = type(n)
        if t is ast.Sym:
            if n.name not in scoped:
                free.add(n.name)
            return
        if t is ast.Local:
            for v in n.values:
                scan(v)
            scoped.update(n.names)
            return
        if t is ast.NumericFor:
            scoped.add(n.var)
        elif t is ast.GenericFor:
            scoped.update(n.names)
        elif t in (ast.FuncDef, ast.FuncExpr):
            scoped.update(n.params)
        if t is ast.Field:
            scan(n.base)          # `.name` is a literal key, not a read
            return
        for child in _walk(n):
            scan(child)

    scan(node)
    return free


def _contains(node, kinds: tuple) -> bool:
    if type(node) in kinds:
        return True
    return any(_contains(c, kinds) for c in _walk(node))


def collect_runtime_written(load_sources, command_sources, parse) -> tuple:
    """Global names something can rebind WHILE THE SWEEP RUNS, plus whether the
    chart writes globals dynamically at all.

    A symbol may only be cached across ticks if nothing can change what it
    means mid-run. `STORE_GLOBAL` already evicts the names the compiled body
    itself writes; this covers everyone ELSE sharing the namespace, and it is a
    static question, so it is answered statically.

    Runtime, in this sim, means:
    - an assignment lexically inside ANY function body, load chunk or not. A
      top-level `function f() X = 1 end` writes `X` whenever `f` is called, not
      when the chunk ran.
    - anything a COMMAND body assigns. Command bodies re-fire for the whole
      song (an Idle chain re-queues itself), and one cached chunk is re-run
      with `self` rebound per firing actor - so a shared `X = self` rebinds X
      per fire.
    A load chunk's TOP-LEVEL assignment is not runtime: chunks complete before
    the sweep starts. That is what keeps `mod_firstSeenBeat` eligible.

    Returns `(names, dynamic)`. `dynamic` is True when any chunk assigns through
    a computed key (`_G[expr] = v`), which no name-keyed analysis can track -
    the caller must then trust nothing. Chart Lua is arbitrary user content, so
    this is the escape hatch that keeps the rule sound rather than merely
    unfalsified on the charts to hand."""
    names: set = set()
    dynamic = False

    def targets(node, inside_function):
        nonlocal dynamic
        kind = type(node)
        if kind in (ast.FuncDef, ast.FuncExpr):
            for child in _walk(node):
                targets(child, True)
            return
        if kind is ast.Assign:
            for target in node.targets:
                if type(target) is ast.Sym:
                    if inside_function:
                        names.add(target.name)
                elif type(target) is ast.Index:
                    base = target.base
                    if type(base) is ast.Sym and base.name == '_G':
                        dynamic = True
        for child in _walk(node):
            targets(child, inside_function)

    for source in load_sources:
        try:
            stmts, _sink = parse(source)
        except Exception:
            continue
        for stmt in stmts:
            targets(stmt, False)
    for source in command_sources:
        try:
            stmts, _sink = parse(source)
        except Exception:
            continue
        for stmt in stmts:
            # Every level of a command body is runtime - it re-fires.
            targets(stmt, True)
    return names, dynamic


def collect_inlinable_helpers(chunk_sources, parse) -> dict:
    """Top-level `function NAME(params) ... end` definitions that may be INLINED
    into another body at their call sites.

    A chart's per-frame body calls small helpers that live in the chart's Lua
    (`perframe(a, b)` is 35 calls per tick on gat, 82% of all free calls). Each
    one is a full crossing out to the host just to run a few lines of
    arithmetic. Inlining compiles those lines into the op stream instead, where
    the clock the helper reads is already resolved.

    The definition STAYS in the host, untouched: this mints a second, compiled
    copy for the compiled body's own call sites. Two copies of a function are
    only safe when the function has no identity and no private state, which is
    what the screen below enforces:

    - no `FuncDef`/`FuncExpr` in the body - a closure minted by the inlined copy
      could be stored and later invoked by the host, and a closure created in
      one interpreter cannot be called by another.
    - no free name that is a LOCAL of the defining chunk. A top-level function
      may close over a chunk-local (`local __C = ...` then `function f() __C[k]
      end`); inlined, that name would compile to a global read and silently
      resolve to nothing. This is the screen that matters most, and it is why
      the chunk's own locals have to be collected rather than just the body's.
    - no varargs, and no recursion into another inlinable helper (the inliner
      refuses to nest, so a self-call falls back to the ordinary crossing).

    Side effects are NOT screened out: an inlined body performs exactly the
    operations the host copy would, in the same order, against the same shared
    globals. It is state and identity that cannot be duplicated, not effects.

    `chunk_sources` is an iterable of Lua source strings; `parse` turns one into
    a statement tuple. A name defined more than once with differing bodies is
    dropped - which definition wins at runtime is load-order, not ours to
    guess."""
    found: dict = {}
    rejected: set = set()
    for source in chunk_sources:
        try:
            stmts, _sink = parse(source)
        except Exception:
            continue
        chunk_locals: set = set()
        for stmt in stmts:
            if type(stmt) is ast.Local:
                chunk_locals.update(stmt.names)
        for stmt in stmts:
            if type(stmt) is not ast.FuncDef or stmt.is_local:
                continue
            name = stmt.name
            if _contains_body(stmt, (ast.FuncDef, ast.FuncExpr)):
                rejected.add(name)
                continue
            free = set()
            for s in stmt.body:
                free |= _free_names(s, set(stmt.params))
            if free & chunk_locals:
                rejected.add(name)
                continue
            if len(stmt.body) > _INLINE_MAX_STMTS \
                    or _inline_crossing_cost(stmt) > _INLINE_MAX_CROSSINGS:
                rejected.add(name)
                continue
            prior = found.get(name)
            if prior is not None and prior != stmt:
                rejected.add(name)
                continue
            found[name] = stmt
    return {k: v for k, v in found.items() if k not in rejected}


def _inline_crossing_cost(funcdef) -> int:
    """Frontier crossings the helper would emit AT AN INLINED CALL SITE.

    Inlining moves the helper's work out of the host and into the op stream,
    where every host read becomes a crossing that cost nothing while it ran
    inside the host. So a helper pays for itself only when its body crosses
    less than the call it replaces. Decided once per definition at compile
    time, never sampled per crossing.

    Compiled through the REAL inline path, not the body in isolation: compiled
    standalone, the parameters and locals are unbound and turn into global
    reads and stores, and `return <value>` becomes a FALLBACK - which charged
    gat's 3-crossing `perframe` at over twice that and rejected it.

    gat's `perframe` costs 3 (two clock getters the executor answers itself,
    plus one global read); `do back burn` ships a different `perframe` costing
    far more, and inlining that one measured +3.4%."""
    compiler = Compiler()
    compiler._global_writes = collect_global_writes(funcdef.body)
    compiler._inline_fns = {funcdef.name: funcdef}
    scope = _Scope()
    compiler._slot(scope, 'self')
    call = ast.Call(fn=ast.Sym(name=funcdef.name),
                    args=tuple(ast.Nil() for _ in funcdef.params))
    try:
        compiler._expr(call, scope)
    except Exception:
        return _INLINE_MAX_CROSSINGS + 1      # uncompilable: do not inline
    cost = 0
    for op, a, _b, _c in compiler.ops:
        kind = _OP_NAMES.get(op)
        if kind not in _CROSSING_OPS:
            continue
        if kind == 'GETTER' and a < len(compiler.names) \
                and compiler.names[a] in _CLOCK_VERBS:
            continue
        cost += 1
    return cost


def _contains_body(funcdef, kinds: tuple) -> bool:
    return any(_contains(s, kinds) for s in funcdef.body)


# How deep helper inlining may nest before falling back to a crossing, and the
# largest helper body worth duplicating at every call site.
_INLINE_DEPTH = 4
_INLINE_MAX_STMTS = 12

# Ops that leave the executor for the host. Inlining a helper only pays if its
# body emits FEWER of these than the single CALL_SYM it replaces costs - and a
# crossing is worth several ops, hence a budget rather than one-for-one.
_CROSSING_OPS = frozenset({
    'GETTER', 'METHOD', 'POKE', 'CALL_SYM', 'CALL_VALUE', 'LOAD_SYMBOL',
    'LOAD_GLOBAL', 'STORE_GLOBAL', 'INDEX', 'SET_INDEX', 'FALLBACK',
    'TABLE_INSERT'})
# The executor answers these from its own per-tick clock, so a getter on one
# never reaches the host and must not be charged to the helper.
_CLOCK_VERBS = frozenset({'GetSongBeat', 'GetSongTime'})
_INLINE_MAX_CROSSINGS = 6


# Opcode value -> name, for the crossing-cost accounting above.
_OP_NAMES = {int(getattr(Op, a)): a for a in dir(Op) if a.isupper()}


class ReceiverSources:
    """Which crossing-OUT sources can reach a METHOD RECEIVER position.

    The frontier can tag a value as a live actor on its way out, which makes a
    later `recv:verb(...)` on it cheap. But the tag costs a probe on EVERY value
    that site emits, while it only pays back on values that come BACK as a
    receiver - and on a real chart that ratio ran about 2:1 against, which is
    why sampling actor density at runtime measured net-zero.

    Whether a value can ever be a receiver is a STATIC property of the body, so
    it is decided here instead. `symbols` / `calls` / `getters` name the sources
    whose result can flow into a receiver; `index` / `field` are whole-site
    flags because those crossings carry no name to key on.

    Flow is traced through variable bindings to a fixed point, so the common
    `local p = Plr(1)  p:zoom(x)` idiom marks `Plr` even though the receiver is
    a slot. Conservative in the direction of MORE probing: an unrecognised
    receiver shape turns its site on rather than off, because a site wrongly on
    only costs speed while a site wrongly off would tag nothing and lose the
    identity the executor depends on."""

    __slots__ = ('symbols', 'calls', 'getters', 'index', 'field')

    def __init__(self):
        self.symbols: set[str] = set()
        self.calls: set[str] = set()
        self.getters: set[str] = set()
        self.index = False
        self.field = False


def _walk(node):
    """Every child Node of `node`, through tuple fields."""
    for f in getattr(node, '__dataclass_fields__', {}):
        v = getattr(node, f)
        if isinstance(v, ast.Node):
            yield v
        elif isinstance(v, tuple):
            for item in v:
                if isinstance(item, ast.Node):
                    yield item


def collect_receiver_sources(stmts) -> ReceiverSources:
    """See `ReceiverSources`. Walks the body once to collect every receiver
    expression and every name binding, then propagates receiver-ness backwards
    through the bindings until it stops growing."""
    found = ReceiverSources()
    bindings: dict = {}
    pending: list = []

    def scan(node):
        t = type(node)
        if t is ast.Method:
            pending.append(node.recv)
        elif t is ast.Assign:
            for target, value in zip(node.targets, node.values):
                if type(target) is ast.Sym:
                    bindings.setdefault(target.name, []).append(value)
        elif t is ast.Local:
            for name, value in zip(node.names, node.values):
                bindings.setdefault(name, []).append(value)
        for child in _walk(node):
            scan(child)

    for stmt in stmts:
        scan(stmt)

    seen_names: set = set()
    while pending:
        node = pending.pop()
        t = type(node)
        if t is ast.Sym:
            # A receiver read by NAME: the symbol site must tag it, and so must
            # whatever the name was bound from (one hop per iteration).
            found.symbols.add(node.name)
            if node.name not in seen_names:
                seen_names.add(node.name)
                pending.extend(bindings.get(node.name, ()))
        elif t is ast.Call:
            if type(node.fn) is ast.Sym:
                found.calls.add(node.fn.name)
            else:
                pending.append(node.fn)
        elif t is ast.Method:
            found.getters.add(node.name)
        elif t is ast.Index:
            found.index = True
        elif t is ast.Field:
            found.field = True
        elif t is not None:
            # An unrecognised receiver shape (a parenthesised expression, a
            # binary result): recurse rather than silently declining to tag.
            pending.extend(_walk(node))
    return found


def collect_global_writes(stmts) -> set:
    """Names ASSIGNED anywhere in the body but never declared `local` / a loop
    var / param at that point - Lua implicit globals = the accumulator set.
    Conservative: a name that is EVER a local anywhere is treated as local-
    capable (the slot resolver decides per-use-site); this set is only the
    fallback classification for a free Sym in `_sym`."""
    writes: set = set()
    locals_seen: set = set()

    def scan(node):
        t = type(node)
        if t is ast.Local:
            locals_seen.update(node.names)
            for v in node.values:
                scan(v)
            return
        if t is ast.Assign:
            for v in node.values:
                scan(v)
            for tgt in node.targets:
                if type(tgt) is ast.Sym:
                    writes.add(tgt.name)
                else:
                    scan(tgt)
            return
        if t in (ast.NumericFor, ast.GenericFor):
            locals_seen.update([node.var] if t is ast.NumericFor else node.names)
        for f in getattr(node, '__dataclass_fields__', {}):
            v = getattr(node, f)
            if isinstance(v, (list, tuple)):
                for x in v:
                    if hasattr(x, '__dataclass_fields__'):
                        scan(x)
                    elif isinstance(x, tuple):
                        for y in x:
                            if hasattr(y, '__dataclass_fields__'):
                                scan(y)
            elif hasattr(v, '__dataclass_fields__'):
                scan(v)

    for s in stmts:
        scan(s)
    # a name both assigned and ever-local: the per-site slot resolver wins, so
    # keep it OUT of the pure-global set to avoid a global shadowing a local.
    return writes - locals_seen


def compile_body_ops(stmts, inline_fns=None, prop_gets=None,
                     prop_sets=None):
    """Compile a parsed body to an OpProgram. Entry point."""
    writes = collect_global_writes(stmts)
    receivers = collect_receiver_sources(stmts)
    return Compiler().compile(stmts, writes, receivers, inline_fns,
                              prop_gets, prop_sets)
