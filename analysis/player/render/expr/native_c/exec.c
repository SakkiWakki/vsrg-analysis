/* See exec.h. The computed-goto op-stream trampoline. Opcode numbers MUST match
 * opstream.py::Op. Semantics mirror frame_compile_exec.py op-for-op. */
#include "exec.h"
#include "cvalue_ops.h"

#include <stdlib.h>
#include <string.h>

/* opcodes - keep in sync with opstream.py::Op (and _NUMFOR_TEST/STEP = 30/31) */
enum {
    OP_CONST=0, OP_LOAD_SLOT, OP_STORE_SLOT, OP_LOAD_GLOBAL, OP_STORE_GLOBAL,
    OP_LOAD_SYMBOL, OP_BINARY, OP_UNARY, OP_INDEX, OP_FIELD, OP_GETTER,
    OP_METHOD, OP_CALL_SYM, OP_CALL_MATH, OP_CALL_BUILTIN, OP_MAKE_TABLE,
    OP_NEWTABLE_ARR, OP_POKE, OP_SET_INDEX, OP_SET_FIELD, OP_POP, OP_JUMP,
    OP_JUMP_IF_FALSE, OP_JIF_FALSE_KEEP, OP_JIF_TRUE_KEEP, OP_DUP,
    OP_RETURN_HALT, OP_ITER_SETUP, OP_ITER_NEXT, OP_FALLBACK,
    OP_NUMFOR_TEST, OP_NUMFOR_STEP, OP_TABLE_INSERT, OP_CALL_VALUE,
    OP__COUNT
};

/* register stack helpers */
#define PUSH(v) (st->regs[sp++] = (v))
#define POP()   (st->regs[--sp])
#define PEEK()  (st->regs[sp-1])

/* math fn ids match opstream._MATH_FNS order */
#include <math.h>

/* M_PI is not in ISO C (only POSIX/GNU math.h); define it under -std=c11
 * where the feature-test macros aren't set, so the build doesn't depend on
 * the compiler's default dialect. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static CValue call_math(int fn, const CValue *a, int argc) {
    /* all math fns take numbers; a non-number arg -> UNRESOLVED (Lua/frame_eval
     * would error -> our try/except -> UNRESOLVED). */
    double x = argc > 0 && cv_is_num(a[0]) ? cv_as_num(a[0]) : 0.0;
    double y = argc > 1 && cv_is_num(a[1]) ? cv_as_num(a[1]) : 0.0;
    for (int i = 0; i < argc; i++) if (!cv_is_num(a[i])) return cv_unresolved();
    double r;
    switch (fn) {
        case 0:  r = sin(x); break;   case 1:  r = cos(x); break;
        case 2:  r = tan(x); break;   case 3:  r = asin(x); break;
        case 4:  r = acos(x); break;  case 5:  r = atan(x); break;
        case 6:  r = atan2(x, y); break; case 7: r = sinh(x); break;
        case 8:  r = cosh(x); break;  case 9:  r = tanh(x); break;
        case 10: r = exp(x); break;   case 11: r = log(x); break;
        case 12: r = log10(x); break; case 13: r = sqrt(x); break;
        case 14: r = fabs(x); break;  case 15: r = floor(x); break;
        case 16: r = ceil(x); break;  case 17: r = fmod(x, y); break;
        case 18: r = pow(x, y); break; case 19: r = x * 180.0 / M_PI; break;
        case 20: r = x * M_PI / 180.0; break;
        case 21: r = (x < y) ? x : y; break;   /* min */
        case 22: r = (x > y) ? x : y; break;   /* max */
        case 23: return cv_unresolved();       /* random: non-deterministic, unmodeled */
        default: return cv_unresolved();
    }
    if (isnan(r)) return cv_unresolved();
    return cv_num(r);
}

int cexec_run(CExecState *st, CValue self_val) {
    /* clear locals, rebind self */
    for (int i = 0; i < st->nslots; i++) st->frame[i] = cv_nil();
    st->frame[0] = self_val;
    int sp = 0;
    const COp *ops = st->ops;
    CFrontier *fe = st->fe;
    int pc = 0;

#if defined(__GNUC__) || defined(__clang__)
    /* direct-threaded dispatch */
    static void *const DISPATCH[OP__COUNT] = {
        &&L_CONST, &&L_LOAD_SLOT, &&L_STORE_SLOT, &&L_LOAD_GLOBAL, &&L_STORE_GLOBAL,
        &&L_LOAD_SYMBOL, &&L_BINARY, &&L_UNARY, &&L_INDEX, &&L_FIELD, &&L_GETTER,
        &&L_METHOD, &&L_CALL_SYM, &&L_CALL_MATH, &&L_CALL_BUILTIN, &&L_MAKE_TABLE,
        &&L_NEWTABLE_ARR, &&L_POKE, &&L_SET_INDEX, &&L_SET_FIELD, &&L_POP, &&L_JUMP,
        &&L_JUMP_IF_FALSE, &&L_JIF_FALSE_KEEP, &&L_JIF_TRUE_KEEP, &&L_DUP,
        &&L_RETURN_HALT, &&L_ITER_SETUP, &&L_ITER_NEXT, &&L_FALLBACK,
        &&L_NUMFOR_TEST, &&L_NUMFOR_STEP, &&L_TABLE_INSERT, &&L_CALL_VALUE
    };
    /* NEXT does NOT poll abort - only a frontier call can raise, so CHECK_ABORT
     * is invoked only by the ops that cross (getter/poke/call/index/global/
     * fallback). Polling every op was ~276K ctypes round-trips/tick = the
     * dominant cost. */
    #define NEXT() goto *DISPATCH[ops[pc].op]
    #define CHECK_ABORT() do { if (fe->aborted(fe->ctx)) return 1; } while (0)
    #define OP(name) L_##name:
    goto *DISPATCH[ops[pc].op];
#else
    #define NEXT() goto dispatch
    #define CHECK_ABORT() do { if (fe->aborted(fe->ctx)) return 1; } while (0)
    #define OP(name) case OP_##name:
    dispatch: switch (ops[pc].op) {
#endif

    OP(CONST) {
        PUSH(st->consts[ops[pc].a]); pc++; NEXT();
    }
    OP(LOAD_SLOT) {
        PUSH(st->frame[ops[pc].a]); pc++; NEXT();
    }
    OP(STORE_SLOT) {
        st->frame[ops[pc].a] = POP(); pc++; NEXT();
    }
    OP(LOAD_GLOBAL) {
        PUSH(fe->global_get(fe->ctx, st->names[ops[pc].a])); pc++; NEXT();
    }
    OP(STORE_GLOBAL) {
        fe->global_set(fe->ctx, st->names[ops[pc].a], POP()); pc++; NEXT();
    }
    OP(LOAD_SYMBOL) {
        PUSH(fe->symbol(fe->ctx, st->names[ops[pc].a])); pc++; NEXT();
    }
    OP(BINARY) {
        CValue b = POP(), a = POP();
        PUSH(cv_binary(st->arena, ops[pc].a, a, b)); pc++; NEXT();
    }
    OP(UNARY) {
        CValue a = POP();
        /* #handle (host table length) crosses the frontier; arena tables/strings
         * are handled natively by cv_unary. (op id 2 = CUN_LEN.) */
        if (ops[pc].a == 2 && cv_is_handle(a)) {
            int64_t n = fe->length(fe->ctx, a);
            PUSH(n < 0 ? cv_unresolved() : cv_num((double)n)); pc++; NEXT();
        }
        PUSH(cv_unary(st->arena, ops[pc].a, a)); pc++; NEXT();
    }
    OP(INDEX) {
        CValue key = POP(), base = POP();
        if (cv_is_unresolved(base) || cv_is_unresolved(key)) { PUSH(cv_unresolved()); pc++; NEXT(); }
        if (cv_is_table(base)) PUSH(carena_table_get(st->arena, cv_payload(base), key));
        else PUSH(fe->index(fe->ctx, base, key));   /* host table / handle */
        pc++; NEXT();
    }
    OP(FIELD) {
        CValue base = POP();
        if (cv_is_unresolved(base)) { PUSH(cv_unresolved()); pc++; NEXT(); }
        /* field name as an interned arena string key */
        const char *nm = st->names[ops[pc].a];
        if (cv_is_table(base)) {
            uint64_t sid = carena_intern(st->arena, nm, strlen(nm));
            PUSH(carena_table_get(st->arena, cv_payload(base), cv_str(sid)));
        } else {
            uint64_t sid = carena_intern(st->arena, nm, strlen(nm));
            PUSH(fe->index(fe->ctx, base, cv_str(sid)));
        }
        pc++; NEXT();
    }
    OP(GETTER) {
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        CValue r = fe->getter(fe->ctx, recv, st->names[ops[pc].a], args, argc);
        sp -= argc + 1;
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }
    OP(METHOD) {  /* same shape as GETTER for now (GetChild/GetShader) */
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        CValue r = fe->getter(fe->ctx, recv, st->names[ops[pc].a], args, argc);
        sp -= argc + 1;
        PUSH(r); pc++; NEXT();
    }
    OP(CALL_SYM) {
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue r = fe->call(fe->ctx, st->names[ops[pc].a], args, argc);
        sp -= argc;
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }
    OP(CALL_MATH) {
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue r = call_math(ops[pc].a, args, argc);
        sp -= argc;
        PUSH(r); pc++; NEXT();
    }
    OP(CALL_BUILTIN) {
        /* type/tonumber/tostring - delegate to the frontier's call by name for
         * now (correctness-first; these are low frequency). */
        int argc = ops[pc].b;
        static const char *BN[] = {"type", "tonumber", "tostring"};
        CValue *args = &st->regs[sp - argc];
        CValue r = fe->call(fe->ctx, BN[ops[pc].a], args, argc);
        sp -= argc;
        PUSH(r); pc++; NEXT();
    }
    OP(MAKE_TABLE) {
        int n = ops[pc].a;
        uint64_t tid = carena_table_new_array(st->arena, n ? (size_t)n : 1);
        for (int i = 0; i < n; i++)
            carena_table_seti(st->arena, tid, i + 1, st->regs[sp - n + i]);
        sp -= n;
        PUSH(cv_table(tid)); pc++; NEXT();
    }
    OP(NEWTABLE_ARR) {
        PUSH(cv_table(carena_table_new_array(st->arena, (size_t)ops[pc].a)));
        pc++; NEXT();
    }
    OP(POKE) {
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        fe->poke(fe->ctx, recv, st->names[ops[pc].a], args, argc);
        sp -= argc + 1;
        CHECK_ABORT();
        pc++; NEXT();
    }
    OP(SET_INDEX) {
        /* stack (bottom->top): value, base, key  (assign pushed value first) */
        CValue key = POP(), base = POP(), value = POP();
        if (cv_is_table(base))
            carena_table_set(st->arena, cv_payload(base), key, value);
        else
            fe->set_index(fe->ctx, base, key, value);
        pc++; NEXT();
    }
    OP(SET_FIELD) {
        CValue base = POP(), value = POP();
        const char *nm = st->names[ops[pc].a];
        uint64_t sid = carena_intern(st->arena, nm, strlen(nm));
        if (cv_is_table(base))
            carena_table_set(st->arena, cv_payload(base), cv_str(sid), value);
        else
            fe->set_index(fe->ctx, base, cv_str(sid), value);
        pc++; NEXT();
    }
    OP(POP) { sp--; pc++; NEXT(); }
    OP(JUMP) { pc = ops[pc].a; NEXT(); }
    OP(JUMP_IF_FALSE) {
        CValue c = POP();
        pc = cv_truthy(c) ? pc + 1 : ops[pc].a; NEXT();
    }
    OP(JIF_FALSE_KEEP) {  /* and: if a falsy -> keep a, jump; else pop happens next op */
        CValue c = PEEK();
        if (cv_is_unresolved(c)) { pc = ops[pc].a; NEXT(); }  /* poison: keep, skip */
        pc = cv_truthy_raw(c) ? pc + 1 : ops[pc].a; NEXT();
    }
    OP(JIF_TRUE_KEEP) {   /* or: if a truthy -> keep a, jump */
        CValue c = PEEK();
        if (cv_is_unresolved(c)) { pc = ops[pc].a; NEXT(); }  /* poison: keep, skip */
        pc = cv_truthy_raw(c) ? ops[pc].a : pc + 1; NEXT();
    }
    OP(DUP) { CValue v = PEEK(); PUSH(v); pc++; NEXT(); }
    OP(RETURN_HALT) { return 0; }
    OP(ITER_SETUP) {
        /* ipairs/pairs form: a=mode (0 ipairs, 1 pairs), stack top = the table.
         * iter_setup builds the (k,v) row iterator; returns an opaque id. */
        int mode = ops[pc].a;
        CValue table = POP();
        CValue two[2] = { cv_num((double)mode), table };
        uint64_t it = fe->iter_setup(fe->ctx, two, 2);
        PUSH(cv_handle(it)); pc++; NEXT();
    }
    OP(ITER_NEXT) {
        uint64_t it = cv_payload(PEEK());   /* iterator handle stays on stack */
        int nvars = ops[pc].b;
        CValue vars[8];
        int got = fe->iter_next(fe->ctx, it, vars, nvars < 8 ? nvars : 8);
        if (!got) { sp--; pc = ops[pc].c; NEXT(); }  /* pop iter, exit */
        for (int i = 0; i < nvars; i++) st->frame[ops[pc].a + i] = vars[i];
        pc++; NEXT();
    }
    OP(FALLBACK) {
        CValue r = fe->fallback(fe->ctx, ops[pc].a, st->frame, st->nslots);
        PUSH(r); pc++; NEXT();
    }
    OP(NUMFOR_TEST) {
        /* a=i-slot, b=lim-slot, c=exit-target. The step slot is lim+1 by the
         * consecutive-alloc contract in opstream._numeric_for (the record has
         * no room for a 4th operand). */
        double i = cv_as_num(st->frame[ops[pc].a]);
        double lim = cv_as_num(st->frame[ops[pc].b]);
        double step = cv_as_num(st->frame[ops[pc].b + 1]);
        int cont = step >= 0 ? (i <= lim) : (i >= lim);
        pc = cont ? pc + 1 : ops[pc].c; NEXT();
    }
    OP(NUMFOR_STEP) {
        double i = cv_as_num(st->frame[ops[pc].a]);
        double step = cv_as_num(st->frame[ops[pc].b]);
        st->frame[ops[pc].a] = cv_num(i + step);
        pc++; NEXT();
    }
    OP(TABLE_INSERT) {  /* stack: table, value */
        CValue v = POP(), t = POP();
        if (cv_is_table(t))
            carena_table_append(st->arena, cv_payload(t), v);
        else
            fe->table_insert(fe->ctx, t, v);   /* host table */
        pc++; NEXT();
    }
    OP(CALL_VALUE) {  /* stack: fn, args... */
        int argc = ops[pc].a;
        CValue *args = &st->regs[sp - argc];
        CValue fn = st->regs[sp - argc - 1];
        CValue r = fe->call_value(fe->ctx, fn, args, argc);
        sp -= argc + 1;
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }

#if !(defined(__GNUC__) || defined(__clang__))
    } /* switch */
    goto dispatch;
#endif
}
