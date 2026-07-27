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
    OP_GET_PROP, OP_SET_PROP, OP_CALL_FIELD,
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
    /* clear locals, rebind self. The for-cursor stack resets too: a tick the
     * frontier aborted mid-loop leaves cursors parked, and carrying them into
     * the next tick would resume the wrong table. */
    for (int i = 0; i < st->nslots; i++) st->frame[i] = cv_nil();
    st->frame[0] = self_val;
    st->niter = 0;
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
        &&L_NUMFOR_TEST, &&L_NUMFOR_STEP, &&L_TABLE_INSERT, &&L_CALL_VALUE,
        &&L_GET_PROP, &&L_SET_PROP, &&L_CALL_FIELD
    };
    /* NEXT does NOT poll abort - only a frontier call can raise, so CHECK_ABORT
     * is invoked only by the ops that cross (getter/poke/call/index/global/
     * fallback). Polling every op was ~276K ctypes round-trips/tick = the
     * dominant cost. The check itself is a LOAD of the host's flag, not a
     * crossing of its own - see CFrontier.abort_flag. */
    #define NEXT() goto *DISPATCH[ops[pc].op]
    #define CHECK_ABORT() do { if (*fe->abort_flag) return 1; } while (0)
    #define OP(name) L_##name:
    goto *DISPATCH[ops[pc].op];
#else
    #define NEXT() goto dispatch
    #define CHECK_ABORT() do { if (*fe->abort_flag) return 1; } while (0)
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
        int gid = ops[pc].a;
        CTrim *gtr = st->trim;
        uint8_t *gok = gtr ? gtr->gok : 0;      /* NULL: host cannot invalidate */
        if (gok && gok[gid]) { PUSH(gtr->gval[gid]); pc++; NEXT(); }
        CValue gv = fe->global_get(fe->ctx, st->names[gid]);
        /* An aborted crossing did not answer with the global's value, so the
         * default it left behind must not become the cached one. */
        if (gok && !*fe->abort_flag) { gtr->gval[gid] = gv; gok[gid] = 1; }
        PUSH(gv); pc++; NEXT();
    }
    OP(STORE_GLOBAL) {
        CValue stored = POP();
        fe->global_set(fe->ctx, st->names[ops[pc].a], stored);
        if (st->trim) {
            st->trim->memo_epoch++;
            /* The store reports itself back through the host's write observer,
             * which drops this entry - so the cache is refilled AFTER the
             * crossing, and with the value the store just wrote rather than a
             * read that would have to cross again to learn it. */
            if (st->trim->gok) {
                st->trim->gval[ops[pc].a] = stored;
                st->trim->gok[ops[pc].a] = (uint8_t)(*fe->abort_flag ? 0 : 1);
            }
            /* A name this body STORES is by definition not process-stable, so
             * drop it out of the stable cache permanently (0, not back to 1 -
             * the host probes a name for snapshotting only once, and a name
             * the body writes must never re-promote). Bumping the epoch alone
             * does NOT cover this: LOAD_SYMBOL tests `stable == 2` BEFORE the
             * epoch, so a stable entry would survive its own overwrite and
             * serve the pre-store value for the life of the CBody.
             *
             * Reachable because a name can compile to LOAD_SYMBOL at its reads
             * while STORE_GLOBAL writes it: `collect_global_writes` subtracts
             * every name that is EVER a local anywhere, so a name that is
             * `local` in one scope and an implicit global in another is not in
             * `_global_writes` and `_sym` routes it to LOAD_SYMBOL. Harmless
             * only while such names hold values that never get marked stable. */
            st->trim->stable[ops[pc].a] = 0;
        }
        pc++; NEXT();
    }
    OP(LOAD_SYMBOL) {
        int id = ops[pc].a;
        CTrim *tr = st->trim;
        if (tr && tr->stable[id] == 2) { PUSH(tr->memo_val[id]); pc++; NEXT(); }
        if (tr && tr->memo_gen[id] == tr->memo_epoch) {
            PUSH(tr->memo_val[id]); pc++; NEXT();
        }
        CValue sv = fe->symbol(fe->ctx, st->names[id]);
        if (tr) {
            tr->memo_val[id] = sv; tr->memo_gen[id] = tr->memo_epoch;
            /* The HOST decides what may cache across runs, and only marks a
             * name it can keep valid: an arena snapshot, or a value it boxes
             * per NAME so the CValue is stable (cbody._out_named). Handles are
             * no longer refused here - a pinned name box is a handle whose
             * payload outlives the tick, and refusing it would exclude exactly
             * the singletons this exists for. The host drops every mark via
             * cbody_clear_stable when unseen host code may have rebound. */
            if (tr->stable[id] == 1) tr->stable[id] = 2;
        }
        PUSH(sv); pc++; NEXT();
    }
    OP(BINARY) {
        CValue b = POP(), a = POP();
        PUSH(cv_binary(st->arena, ops[pc].a, a, b)); pc++; NEXT();
    }
    OP(UNARY) {
        CValue a = POP();
        /* #handle (host table length) crosses the frontier; arena tables/strings
         * are handled natively by cv_unary. (op id 2 = CUN_LEN.) An ACTOR is a
         * host table too - without it here `#P1` would stop crossing and become
         * UNRESOLVED instead of the 0 the host answers. */
        if (ops[pc].a == 2 && (cv_is_handle(a) || cv_is_actor(a))) {
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
        int vid = ops[pc].a;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        CTrim *tr = st->trim;
        if (tr && tr->clock_recv_set && argc == 0 && recv == tr->clock_recv
                && (vid == tr->clock_beat_id || vid == tr->clock_time_id)) {
            sp -= 1;
            PUSH(vid == tr->clock_beat_id ? tr->clock_beat : tr->clock_time);
            pc++; NEXT();
        }
        CValue r = fe->getter(fe->ctx, recv, st->names[vid], args, argc);
        sp -= argc + 1;
        CHECK_ABORT();
        if (tr && !tr->clock_recv_set && argc == 0 && cv_is_handle(recv)
                && (vid == tr->clock_beat_id || vid == tr->clock_time_id)) {
            tr->clock_recv = recv; tr->clock_recv_set = 1;
        }
        PUSH(r); pc++; NEXT();
    }
    OP(GET_PROP) {
        /* A getter whose verb resolved to a plain property at compile time.
         * Carries an int, not a name: nothing to decode, nothing to route.
         * A settled property on a live actor is answered from the mirror with
         * no crossing at all; a running head tween means the value is being
         * interpolated host-side, so that read still crosses. */
        CValue recv = POP();
        CActorProps *ap = st->props;
        if (ap && cv_is_actor(recv)) {
            uint64_t rec = cv_payload(recv);
            if ((int)rec < ap->nactors && !ap->tweening[rec]) {
                int slot = (int)rec * ap->nprops + ops[pc].a;
                if (ap->present[slot]) {
                    PUSH(cv_num(ap->value[slot])); pc++; NEXT();
                }
            }
        }
        CValue r = fe->get_prop(fe->ctx, recv, ops[pc].a);
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }
    OP(SET_PROP) {
        /* A setter resolved to a property at compile time. Still crosses - the
         * write records a keyframe host-side - but carries an int, so there is
         * no name to decode and no verb table to walk. */
        CValue value = POP();
        CValue recv = POP();
        fe->set_prop(fe->ctx, recv, ops[pc].a, value);
        if (st->trim) st->trim->memo_epoch++;
        CHECK_ABORT();
        pc++; NEXT();
    }
    OP(METHOD) {  /* same shape as GETTER for now (GetChild/GetShader) */
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        CValue r = fe->getter(fe->ctx, recv, st->names[ops[pc].a], args, argc);
        sp -= argc + 1;
        if (st->trim) st->trim->memo_epoch++;
        PUSH(r); pc++; NEXT();
    }
    OP(CALL_SYM) {
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue r = fe->call(fe->ctx, st->names[ops[pc].a], args, argc);
        sp -= argc;
        if (st->trim) st->trim->memo_epoch++;
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
        if (st->trim) st->trim->memo_epoch++;
        CHECK_ABORT();
        pc++; NEXT();
    }
    OP(SET_INDEX) {
        /* stack (bottom->top): value, base, key  (assign pushed value first) */
        CValue key = POP(), base = POP(), value = POP();
        if (cv_is_table(base))
            carena_table_set(st->arena, cv_payload(base), key, value);
        else {
            fe->set_index(fe->ctx, base, key, value);
            if (st->trim) st->trim->memo_epoch++;
        }
        pc++; NEXT();
    }
    OP(SET_FIELD) {
        CValue base = POP(), value = POP();
        const char *nm = st->names[ops[pc].a];
        uint64_t sid = carena_intern(st->arena, nm, strlen(nm));
        if (cv_is_table(base))
            carena_table_set(st->arena, cv_payload(base), cv_str(sid), value);
        else {
            fe->set_index(fe->ctx, base, cv_str(sid), value);
            if (st->trim) st->trim->memo_epoch++;
        }
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
         * An ARENA table walks in-C (see CArenaIter) and leaves the TABLE on
         * the stack as its cursor marker; anything else crosses to the host's
         * iter_setup, which leaves an opaque HANDLE. ITER_NEXT tells them apart
         * by tag. */
        int mode = ops[pc].a;
        CValue table = POP();
        if (cv_is_table(table) && st->niter < CEXEC_ITER_DEPTH) {
            CArenaIter *ai = &st->iters[st->niter++];
            ai->tid = cv_payload(table);
            ai->i = 0;
            ai->n = carena_table_len(st->arena, ai->tid);
            PUSH(table); pc++; NEXT();
        }
        CValue two[2] = { cv_num((double)mode), table };
        uint64_t it = fe->iter_setup(fe->ctx, two, 2);
        PUSH(cv_handle(it)); pc++; NEXT();
    }
    OP(ITER_NEXT) {
        CValue cursor = PEEK();   /* iterator cursor stays on the stack */
        int nvars = ops[pc].b;
        int fill = nvars < 8 ? nvars : 8;
        CValue vars[8];
        int got;
        if (cv_is_table(cursor)) {
            CArenaIter *ai = &st->iters[st->niter - 1];
            got = ai->i < ai->n;
            if (got) {
                ai->i++;
                if (fill > 0) vars[0] = cv_num((double)ai->i);
                if (fill > 1) vars[1] = carena_table_geti(st->arena, ai->tid,
                                                          ai->i);
                for (int i = 2; i < fill; i++) vars[i] = cv_nil();
            } else {
                st->niter--;
            }
        } else {
            got = fe->iter_next(fe->ctx, cv_payload(cursor), vars, fill);
        }
        if (!got) { sp--; pc = ops[pc].c; NEXT(); }  /* pop iter, exit */
        for (int i = 0; i < nvars; i++) st->frame[ops[pc].a + i] = vars[i];
        pc++; NEXT();
    }
    OP(FALLBACK) {
        CValue r = fe->fallback(fe->ctx, ops[pc].a, st->frame, st->nslots);
        if (st->trim) st->trim->memo_epoch++;
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
        else {
            fe->table_insert(fe->ctx, t, v);   /* host table */
            if (st->trim) st->trim->memo_epoch++;
        }
        pc++; NEXT();
    }
    OP(CALL_FIELD) {  /* stack: recv, args...  - `t.f(a)` in ONE crossing */
        int argc = ops[pc].b;
        CValue *args = &st->regs[sp - argc];
        CValue recv = st->regs[sp - argc - 1];
        CValue r;
        if (cv_is_unresolved(recv)) {
            r = cv_unresolved();
        } else if (cv_is_table(recv)) {
            /* An ARENA table resolves its own field, exactly as FIELD did. A
             * snapshot never holds a function, but a table the BODY built can
             * (`t.f = some_host_fn`), and only a handle is callable at all. */
            const char *fld = st->names[ops[pc].a];
            CValue v = carena_table_get(
                st->arena, cv_payload(recv),
                cv_str(carena_intern(st->arena, fld, strlen(fld))));
            r = cv_is_handle(v) ? fe->call_value(fe->ctx, v, args, argc)
                                : cv_unresolved();
        } else {
            r = fe->call_field(fe->ctx, recv, st->names[ops[pc].a], args, argc);
        }
        if (st->trim) st->trim->memo_epoch++;
        sp -= argc + 1;
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }
    OP(CALL_VALUE) {  /* stack: fn, args... */
        int argc = ops[pc].a;
        CValue *args = &st->regs[sp - argc];
        CValue fn = st->regs[sp - argc - 1];
        CValue r = fe->call_value(fe->ctx, fn, args, argc);
        if (st->trim) st->trim->memo_epoch++;
        sp -= argc + 1;
        CHECK_ABORT();
        PUSH(r); pc++; NEXT();
    }

#if !(defined(__GNUC__) || defined(__clang__))
    } /* switch */
    goto dispatch;
#endif
}
