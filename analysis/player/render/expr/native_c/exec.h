/* The computed-goto op-stream executor - the C compute core.
 *
 * Runs one compiled Update body (a flat op array) per tick with NO per-node
 * interpretation: a program counter walks the ops, direct-threaded dispatch
 * (computed goto).
 * Values are NaN-boxed CValues; tables/strings live in the kernel arena; the
 * only per-tick crossings back to the host are the frontier vtable calls
 * (getter/poke/symbol/global for live SimActor state), ~3.4K/tick measured.
 *
 * The point is to not rely on any Python for execution during playback.
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
    /* recv's PROPERTY `prop_id` - a getter whose verb resolved to a plain
     * property at compile time, so no name crosses and the host does no verb
     * routing. Same answer as `getter` would give for that verb. */
    CValue (*get_prop)(void *ctx, CValue recv, int prop_id);
    /* recv's PROPERTY `set_id` = value - a setter whose verb resolved to a
     * single property at compile time. Same effect as `poke` with that verb. */
    void   (*set_prop)(void *ctx, CValue recv, int set_id, CValue value);
    /* recv:verb(args) in EFFECT position - a poke (self:zoom(x)). */
    void   (*poke)(void *ctx, CValue recv, const char *verb, const CValue *args, int argc);
    /* a free call name(args) the core did not service. */
    CValue (*call)(void *ctx, const char *name, const CValue *args, int argc);
    /* call a COMPUTED callable value (a[3](x) / a local closure). */
    CValue (*call_value)(void *ctx, CValue fn, const CValue *args, int argc);
    /* base.name(args) - the field read and the call it feeds, together. Split
     * across `index` + `call_value` this was TWO crossings for one call site,
     * and the intermediate never existed for any other purpose. */
    CValue (*call_field)(void *ctx, CValue base, const char *name,
                         const CValue *args, int argc);
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
    /* Host-owned: nonzero once a frontier call has raised (aborts the tick,
     * matching the interp). A FLAG the executor loads, not a callback it
     * calls - the four crossings that can raise polled it after every one
     * (85x/tick on gat, 2.2M over the chart) and the answer is a constant 0
     * for any surface that does not model aborting. The host clears it per
     * run and sets it from the crossing that raised. */
    const int *abort_flag;
} CFrontier;

/* Crossing trim: per-run symbol memo + cross-run stable cache + the
 * clock-getter fast path. Symbols are stable between HOST-MUTATING
 * crossings (poke/method/call/fallback/global-store), so LOAD_SYMBOL
 * re-resolves only after one; `stable` entries (engine singletons,
 * snapshotted tables - flagged 1 by the host, promoted to 2 on first
 * resolution) never re-cross. GETTER of the clock verbs on the learned
 * GAMESTATE handle returns host-preloaded values. */
typedef struct {
    CValue   *memo_val;             /* [nnames] */
    uint32_t *memo_gen;
    uint32_t  memo_epoch;
    uint8_t  *stable;               /* 0 no, 1 wanted, 2 cached */
    /* The accumulator globals, cached by name id. Sound for a different reason
     * than `stable`: the sandbox REPORTS every write by name (a proxied
     * __newindex, rawset included), so the host drops an entry the moment its
     * value can change - and STORE_GLOBAL writes its own value straight in,
     * since the store IS the new value. A global read then costs a load rather
     * than a crossing that walks back into the Lua namespace. */
    CValue   *gval;                 /* [nnames] */
    uint8_t  *gok;
    int clock_beat_id, clock_time_id;   /* name-pool verb ids, -1 unset */
    CValue clock_recv;  int clock_recv_set;
    CValue clock_beat, clock_time;
} CTrim;

/* A generic-for over an ARENA table walked entirely in the executor. A
 * snapshotted (or body-built) arena table is a dense pure array, so ipairs and
 * pairs agree on the 1..n run and there is nothing for the host to decide -
 * crossing per ROW instead was what made the op-stream body SLOWER than the
 * Lua one it replaces (gat iterates ~2.2K rows/tick, each a Python generator
 * resume plus two ctypes calls). Loops nest, so the cursors form a stack; a
 * for over a HOST table still crosses through `iter_setup`/`iter_next`. */
#define CEXEC_ITER_DEPTH 16

typedef struct { uint64_t tid; int64_t i, n; } CArenaIter;

/* Settled actor properties, mirrored from the host so a property read need not
 * cross at all.
 *
 * Only the SETTLED value is here. A read while the actor's head tween is
 * running is a live interpolation off that tween (SimActor.get lerps between
 * the ease-from and the queue tail), which would mean mirroring the whole queue
 * - so those keep crossing, flagged per actor. Measured on gat: 98.4% of actor
 * reads are settled, 1.1% interpolate.
 *
 * Indexed [rec_id * nprops + prop_id]. `present` distinguishes "never written"
 * (the host answers with the property's rest value, which the executor does not
 * know) from a real 0.0.
 *
 * DOUBLE, not float: the host's values are Python floats and the recorded
 * keyframes compare exactly. Storing them at single precision rounded every
 * mirrored read and diverged the keyframe stream - the mirror has to be a
 * mirror, not an approximation. */
typedef struct {
    double  *value;      /* [nactors * nprops] */
    uint8_t *present;    /* [nactors * nprops] */
    uint8_t *tweening;   /* [nactors] - head tween running: read must cross */
    int nactors, nprops;
} CActorProps;

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
    CTrim       *trim;                    /* NULL = trim disabled */
    CArenaIter   iters[CEXEC_ITER_DEPTH]; int niter;  /* in-C for cursors */
    CActorProps *props;                   /* NULL = every read crosses */
} CExecState;

/* Run the body once (one tick). `self_val` is rebound into slot 0; other slots
 * are cleared to nil. Returns 0 on normal completion, 1 if the frontier aborted
 * (the host records the fault). */
int cexec_run(CExecState *st, CValue self_val);

#endif /* NOTITG_EXEC_H */
