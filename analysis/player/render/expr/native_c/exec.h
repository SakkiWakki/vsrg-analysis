/* The computed-goto op-stream executor - the C compute core.
 *
 * Runs one compiled Update body (a flat op array) per tick with NO per-node
 * interpretation: a program counter walks the ops, direct-threaded dispatch
 * (computed goto where the compiler supports it, switch fallback otherwise).
 * Values are NaN-boxed CValues; tables/strings live in the kernel arena; the
 * only per-tick crossings back to the host are the frontier vtable calls
 * (getter/poke/symbol/global for live SimActor state), ~3.4K/tick measured.
 *
 * Trust model: the op array + const pool are built by the validated upstream
 * Python compiler (opstream.py). The executor assumes well-formed input (in-
 * range slot/const/name ids, balanced stack, valid jump targets); it does not
 * re-sanitize. A debug build asserts these invariants.
 */
#ifndef NOTITG_EXEC_H
#define NOTITG_EXEC_H

#include <stdint.h>
#include "cvalue.h"
#include "carena.h"

/* One op record: opcode + three int operands (matches opstream serialize). */
typedef struct { int32_t op, a, b, c; } COp;

/* The frontier vtable: the host (Python) implements these; the executor calls
 * them for everything it does not own. `ctx` is an opaque host pointer. Values
 * cross as CValues (NaN-boxed); a string CValue's payload is an arena id the
 * host resolves via `arena`. */
typedef struct CFrontier {
    void   *ctx;
    /* a bare symbol not a local/global: driver clock (beat), actor global,
     * host fn. UNRESOLVED if unknown. */
    CValue (*symbol)(void *ctx, const char *name);
    /* read/write a body-written global (the accumulator store). */
    CValue (*global_get)(void *ctx, const char *name);
    void   (*global_set)(void *ctx, const char *name, CValue v);
    /* recv:verb(args) in VALUE position - a live getter (self:GetX()). */
    CValue (*getter)(void *ctx, CValue recv, const char *verb, const CValue *args, int argc);
    /* recv:verb(args) in EFFECT position - a poke (self:zoom(x)). */
    void   (*poke)(void *ctx, CValue recv, const char *verb, const CValue *args, int argc);
    /* a free call name(args) the core did not service. */
    CValue (*call)(void *ctx, const char *name, const CValue *args, int argc);
    /* call a COMPUTED callable value (a[3](x) / a local closure). */
    CValue (*call_value)(void *ctx, CValue fn, const CValue *args, int argc);
    /* host_table[key] read (recv is a HANDLE the frontier owns). */
    CValue (*index)(void *ctx, CValue base, CValue key);
    /* host_table[key] = v. */
    void   (*set_index)(void *ctx, CValue base, CValue key, CValue v);
    /* #handle / table.getn(handle) - length of a host table. -1 if not one. */
    int64_t (*length)(void *ctx, CValue base);
    /* table.insert(handle, v) into a host table. */
    void   (*table_insert)(void *ctx, CValue base, CValue v);
    /* generic-for over exprs: returns an opaque iterator id, or 0. */
    uint64_t (*iter_setup)(void *ctx, const CValue *exprs, int n);
    /* advance iterator: fills vars[0..nvars), returns 1 if a row, 0 at end. */
    int    (*iter_next)(void *ctx, uint64_t iter, CValue *vars, int nvars);
    /* run a FALLBACK AST node (by pool id) through the Python interpreter, in
     * this executor's variable environment; returns the node's value (or nil
     * for a statement). frame/globals are shared via the exec state the host
     * captured. */
    CValue (*fallback)(void *ctx, int node_id, CValue *frame, int nslots);
    /* whether a frontier call raised (aborts the tick, matching the interp). */
    int    (*aborted)(void *ctx);
} CFrontier;

/* Execution state for one body, persisted across ticks (the frame's globals /
 * accumulators live host-side behind the frontier; the slot frame is cleared
 * each tick except slot 0 = self). */
typedef struct {
    const COp   *ops;   int nops;
    const CValue *consts; int nconsts;   /* pre-materialized const CValues */
    const char **names;  int nnames;      /* interned name strings */
    CArena      *arena;
    CFrontier   *fe;
    CValue      *frame;  int nslots;      /* slot array (self at [0]) */
    CValue      *regs;   int reg_cap;     /* eval register stack */
} CExecState;

/* Run the body once (one tick). `self_val` is rebound into slot 0; other slots
 * are cleared to nil. Returns 0 on normal completion, 1 if the frontier aborted
 * (the host records the fault). */
int cexec_run(CExecState *st, CValue self_val);

#endif /* NOTITG_EXEC_H */
