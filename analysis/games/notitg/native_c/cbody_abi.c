/* The ctypes ABI for driving the op-stream executor from Python.
 *
 * Python (opstream.py -> serialize()) builds the op array + const pool + names;
 * this ABI materializes them into a CExecState (interning const strings into the
 * arena, pre-boxing const CValues) and runs one tick against a CFrontier whose
 * function pointers are Python ctypes callbacks (the live SimActor getter/poke,
 * the accumulator globals, the FALLBACK interpreter).
 *
 * CValues cross the ctypes boundary as raw uint64 (the NaN-box). A string
 * CValue's payload is an arena id; the boundary provides cbody_str() so Python
 * can resolve one, and cbody_intern() so Python can hand a string back. Numbers/
 * bool/nil/handle need no arena. This keeps the hot per-tick path (numeric slots
 * + arithmetic) entirely in C; only getter/poke/global/fallback cross.
 */
#include "exec.h"
#include "cvalue.h"
#include "carena.h"

#include <stdlib.h>
#include <string.h>

/* A compiled body session: owns the arena, the op array, const/name pools, and
 * the reusable frame/reg buffers. Persists across ticks. */
typedef struct {
    CArena   *arena;
    COp      *ops;    int nops;
    CValue   *consts; int nconsts;
    char    **names;  int nnames;
    CFrontier fe;                  /* filled by Python (callback ptrs + ctx) */
    CValue   *frame;  int nslots;
    CValue   *regs;   int reg_cap;
} CBody;

/* --- lifecycle ---------------------------------------------------------- */

CBody *cbody_new(int nslots, int reg_cap) {
    CBody *b = calloc(1, sizeof(CBody));
    b->arena = carena_new();
    b->nslots = nslots;
    b->frame = calloc(nslots, sizeof(CValue));
    b->reg_cap = reg_cap;
    b->regs = calloc(reg_cap, sizeof(CValue));
    return b;
}

void cbody_free(CBody *b) {
    if (!b) return;
    carena_free(b->arena);
    free(b->ops);
    free(b->consts);
    for (int i = 0; i < b->nnames; i++) free(b->names[i]);
    free(b->names);
    free(b->frame);
    free(b->regs);
    free(b);
}

/* --- load the compiled program ------------------------------------------ */

/* ops: packed int32[4*nops] (opcode,a,b,c). Copied in. */
void cbody_set_ops(CBody *b, const int32_t *ops_flat, int nops) {
    b->nops = nops;
    b->ops = malloc((size_t)nops * sizeof(COp));
    memcpy(b->ops, ops_flat, (size_t)nops * sizeof(COp));
}

/* consts: parallel arrays kinds[nc] (0 num,1 str,2 true,3 false,4 nil) and
 * nums[nc] (num value, or for str the index into the str-consts handed via
 * cbody_set_const_str). Pre-box each into a CValue. */
void cbody_set_consts(CBody *b, const uint8_t *kinds, const double *nums, int nc) {
    b->nconsts = nc;
    b->consts = malloc((size_t)nc * sizeof(CValue));
    for (int i = 0; i < nc; i++) {
        switch (kinds[i]) {
            case 0: b->consts[i] = cv_num(nums[i]); break;
            case 2: b->consts[i] = cv_bool(1); break;
            case 3: b->consts[i] = cv_bool(0); break;
            case 4: b->consts[i] = cv_nil(); break;
            case 1: b->consts[i] = cv_nil(); break; /* filled by set_const_str */
            default: b->consts[i] = cv_nil(); break;
        }
    }
}

/* Set a string const: intern `s` and box it into consts[ci]. */
void cbody_set_const_str(CBody *b, int ci, const char *s, int len) {
    uint64_t id = carena_intern(b->arena, s, (size_t)len);
    b->consts[ci] = cv_str(id);
}

/* names: set one interned name by id (verbs/globals/symbols). */
void cbody_set_names(CBody *b, int nnames) {
    b->nnames = nnames;
    b->names = calloc(nnames, sizeof(char *));
}
void cbody_set_name(CBody *b, int ni, const char *s, int len) {
    b->names[ni] = malloc((size_t)len + 1);
    memcpy(b->names[ni], s, (size_t)len);
    b->names[ni][len] = '\0';
}

/* --- frontier wiring ---------------------------------------------------- */
/* Python passes the callback function pointers + ctx as raw void*. */
void cbody_set_frontier(CBody *b, void *ctx,
        void *symbol, void *gget, void *gset, void *getter, void *poke,
        void *call, void *call_value, void *index, void *set_index, void *length,
        void *table_insert, void *iter_setup, void *iter_next, void *fallback,
        void *aborted) {
    b->fe.ctx = ctx;
    b->fe.symbol     = (CValue(*)(void*,const char*))symbol;
    b->fe.global_get = (CValue(*)(void*,const char*))gget;
    b->fe.global_set = (void(*)(void*,const char*,CValue))gset;
    b->fe.getter     = (CValue(*)(void*,CValue,const char*,const CValue*,int))getter;
    b->fe.poke       = (void(*)(void*,CValue,const char*,const CValue*,int))poke;
    b->fe.call       = (CValue(*)(void*,const char*,const CValue*,int))call;
    b->fe.call_value = (CValue(*)(void*,CValue,const CValue*,int))call_value;
    b->fe.index      = (CValue(*)(void*,CValue,CValue))index;
    b->fe.set_index  = (void(*)(void*,CValue,CValue,CValue))set_index;
    b->fe.length     = (int64_t(*)(void*,CValue))length;
    b->fe.table_insert = (void(*)(void*,CValue,CValue))table_insert;
    b->fe.iter_setup = (uint64_t(*)(void*,const CValue*,int))iter_setup;
    b->fe.iter_next  = (int(*)(void*,uint64_t,CValue*,int))iter_next;
    b->fe.fallback   = (CValue(*)(void*,int,CValue*,int))fallback;
    b->fe.aborted    = (int(*)(void*))aborted;
}

/* --- run one tick ------------------------------------------------------- */
int cbody_run(CBody *b, uint64_t self_val) {
    CExecState st = {0};
    st.ops = b->ops; st.nops = b->nops;
    st.consts = b->consts; st.nconsts = b->nconsts;
    st.names = (const char **)b->names; st.nnames = b->nnames;
    st.arena = b->arena;
    st.fe = &b->fe;
    st.frame = b->frame; st.nslots = b->nslots;
    st.regs = b->regs; st.reg_cap = b->reg_cap;
    return cexec_run(&st, (CValue)self_val);
}

/* --- arena string bridge (for the frontier callbacks) ------------------- */
/* Resolve a string CValue's id to bytes (borrowed). Python reads via ctypes. */
const char *cbody_str(CBody *b, uint64_t str_cv, int *len_out) {
    uint64_t id = cv_payload((CValue)str_cv);
    size_t l; const char *p = carena_str(b->arena, id, &l);
    if (len_out) *len_out = (int)l;
    return p;
}
/* Intern a string, return the boxed str CValue (Python hands a value back). */
uint64_t cbody_intern(CBody *b, const char *s, int len) {
    return (uint64_t)cv_str(carena_intern(b->arena, s, (size_t)len));
}
/* Frame slot access for the FALLBACK bridge (read/write a slot as a CValue). */
uint64_t cbody_frame_get(CBody *b, int slot) { return (uint64_t)b->frame[slot]; }
void     cbody_frame_set(CBody *b, int slot, uint64_t v) { b->frame[slot] = (CValue)v; }

/* Value constructors for Python (so it can build args/results without knowing
 * the box layout). */
uint64_t cbody_num(double d)   { return (uint64_t)cv_num(d); }
uint64_t cbody_nil(void)       { return (uint64_t)cv_nil(); }
uint64_t cbody_true(void)      { return (uint64_t)cv_bool(1); }
uint64_t cbody_false(void)     { return (uint64_t)cv_bool(0); }
uint64_t cbody_unresolved(void){ return (uint64_t)cv_unresolved(); }
uint64_t cbody_handle(uint64_t id) { return (uint64_t)cv_handle(id); }
/* Value inspectors for Python. */
int    cbody_is_num(uint64_t v)   { return cv_is_num((CValue)v); }
int    cbody_is_str(uint64_t v)   { return cv_is_str((CValue)v); }
int    cbody_is_nil(uint64_t v)   { return cv_is_nil((CValue)v); }
int    cbody_is_bool(uint64_t v)  { return cv_is_bool((CValue)v); }
int    cbody_is_handle(uint64_t v){ return cv_is_handle((CValue)v); }
int    cbody_is_unres(uint64_t v) { return cv_is_unresolved((CValue)v); }
int    cbody_is_table(uint64_t v) { return cv_is_table((CValue)v); }
double cbody_as_num(uint64_t v)   { return cv_as_num((CValue)v); }
int    cbody_bool_val(uint64_t v) { return cv_bool_val((CValue)v); }
uint64_t cbody_handle_id(uint64_t v) { return cv_payload((CValue)v); }
