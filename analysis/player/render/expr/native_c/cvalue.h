/* NaN-boxed value model for the NotITG residue executor.
 *
 * The whole point of the C kernel is a table-index-bound interpreter loop where
 * every value is a machine register with zero heap indirection for the common
 * case (numbers). We NaN-box: a `CValue` is a 64-bit word. A real IEEE-754
 * double IS itself (the fast, dominant case - the 608K/tick arithmetic and the
 * 2.1M/tick array indices are numeric); every non-double is smuggled into the
 * payload of a quiet NaN with a 3-bit tag.
 *
 * This mirrors the Rust `native/src/value.rs` enum op-for-op (Num/Str/Bool/Nil/
 * Table/Func/Handle/Unresolved). The load-bearing distinction it must preserve:
 * UNRESOLVED is NOT Nil - Nil is a known Lua absence, UNRESOLVED is the "skip,
 * don't guess" poison sentinel, and `and`/`or` treat them differently. They are
 * two distinct singleton boxes here.
 *
 * Boxing scheme (x86-64 / aarch64, 48-bit pointers/ids) - the LuaJIT/JSC
 * "signalling-NaN with sign bit" trick, chosen so NO real double (incl +-inf
 * and a hardware quiet NaN from 0.0/0.0, which are POSITIVE NaNs, sign=0) can
 * ever collide with a box:
 *   - A boxed non-double sets the SIGN bit AND the full exponent AND the quiet
 *     bit: signature 0xFFF8_0000_0000_0000 in the top 13 bits [63..51]. A normal
 *     double never has exponent-all-ones-with-quiet-bit AND sign set as a NaN
 *     the FPU produces (a negated NaN keeps sign, but our /0->UNRESOLVED and
 *     arith-non-finite->UNRESOLVED upstream rules mean no raw NaN ever flows as
 *     a Num - and even a stray one is positive-signed).
 *   - A double is anything WITHOUT that exact top-13-bit signature - read back
 *     by type-punning u64->f64.
 *   - Below the signature: a 3-bit tag [50..48] then a 48-bit payload [47..0].
 *
 * Payload ids (STR/TABLE/FUNC) index arenas the kernel owns (carena.c); HANDLE
 * carries an opaque host id the frontier resolves. 48 bits is ample (the corpus
 * never approaches 2^48 strings/tables).
 *
 * Trust model: this kernel receives ONLY validated input (a compiler-emitted op
 * array + a deep-copied snapshot arena, both built by upstream Python/Rust). It
 * does no bounds-sanitizing of ids against untrusted data - well-formedness is
 * an upstream invariant. Assertions guard the invariants in debug builds.
 */
#ifndef NOTITG_CVALUE_H
#define NOTITG_CVALUE_H

#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint64_t CValue;

/* Box signature: sign=1, exponent=0x7FF (all ones), quiet bit (51) = 1.
 * Top 13 bits [63..51] = 0xFFF8 >> 3 ... concretely the mask/signature over the
 * full word is 0xFFF8000000000000. A real double (incl a POSITIVE hardware NaN
 * with sign=0) never matches, so cv_is_num is a single masked compare. */
#define CV_SIG       0xFFF8000000000000ULL   /* sign+exp+quiet: the box marker */
#define CV_SIG_MASK  0xFFF8000000000000ULL   /* top 13 bits */
#define CV_TAG_SHIFT 48
#define CV_TAG_MASK  (0x7ULL << CV_TAG_SHIFT) /* bits [50..48] */
#define CV_PAYLOAD_MASK 0x0000FFFFFFFFFFFFULL /* low 48 bits */

/* Tags (3 bits). NIL/TRUE/FALSE/UNRESOLVED are singletons (payload 0);
 * STR/TABLE/FUNC/HANDLE carry a 48-bit id. That is 8 tags - exactly our set,
 * folding Bool into two singleton tags so `truthy` is a pure bit test. */
enum {
    CV_TAG_NIL        = 0,
    CV_TAG_FALSE      = 1,
    CV_TAG_TRUE       = 2,
    CV_TAG_UNRESOLVED = 3,
    CV_TAG_STR        = 4,
    CV_TAG_TABLE      = 5,
    CV_TAG_FUNC       = 6,
    CV_TAG_HANDLE     = 7,
};

/* --- constructors ------------------------------------------------------- */

static inline CValue cv_box(unsigned tag, uint64_t payload) {
    return CV_SIG | ((uint64_t)tag << CV_TAG_SHIFT) | (payload & CV_PAYLOAD_MASK);
}

static inline CValue cv_num(double d) {
    CValue v;
    memcpy(&v, &d, sizeof(v));
    return v;
}

static inline CValue cv_nil(void)        { return cv_box(CV_TAG_NIL, 0); }
static inline CValue cv_unresolved(void) { return cv_box(CV_TAG_UNRESOLVED, 0); }
static inline CValue cv_bool(int b)      { return cv_box(b ? CV_TAG_TRUE : CV_TAG_FALSE, 0); }
static inline CValue cv_str(uint64_t id)    { return cv_box(CV_TAG_STR, id); }
static inline CValue cv_table(uint64_t id)  { return cv_box(CV_TAG_TABLE, id); }
static inline CValue cv_func(uint64_t id)   { return cv_box(CV_TAG_FUNC, id); }
static inline CValue cv_handle(uint64_t id) { return cv_box(CV_TAG_HANDLE, id); }

/* --- predicates --------------------------------------------------------- */

/* A value is a Num iff it is NOT a boxed non-double. A box has the full
 * sign+exp+quiet signature in the top 13 bits; a real double - including +-inf
 * and a POSITIVE hardware NaN (sign=0) - never matches, so every double reads
 * back as Num. (Upstream, /0 and any non-finite arithmetic result become
 * UNRESOLVED, so a negative raw NaN never flows as a Num either.) */
static inline int cv_is_num(CValue v) {
    return (v & CV_SIG_MASK) != CV_SIG;
}

static inline unsigned cv_tag(CValue v) {
    return (unsigned)((v & CV_TAG_MASK) >> CV_TAG_SHIFT);
}

static inline uint64_t cv_payload(CValue v) {
    return v & CV_PAYLOAD_MASK;
}

static inline double cv_as_num(CValue v) {
    double d;
    memcpy(&d, &v, sizeof(d));
    return d;
}

static inline int cv_is_nil(CValue v)        { return !cv_is_num(v) && cv_tag(v) == CV_TAG_NIL; }
static inline int cv_is_unresolved(CValue v) { return !cv_is_num(v) && cv_tag(v) == CV_TAG_UNRESOLVED; }
static inline int cv_is_str(CValue v)        { return !cv_is_num(v) && cv_tag(v) == CV_TAG_STR; }
static inline int cv_is_table(CValue v)      { return !cv_is_num(v) && cv_tag(v) == CV_TAG_TABLE; }
static inline int cv_is_func(CValue v)       { return !cv_is_num(v) && cv_tag(v) == CV_TAG_FUNC; }
static inline int cv_is_handle(CValue v)     { return !cv_is_num(v) && cv_tag(v) == CV_TAG_HANDLE; }
static inline int cv_is_bool(CValue v) {
    if (cv_is_num(v)) return 0;
    unsigned t = cv_tag(v);
    return t == CV_TAG_TRUE || t == CV_TAG_FALSE;
}
static inline int cv_bool_val(CValue v) { return cv_tag(v) == CV_TAG_TRUE; }

/* --- truthiness (value.rs truthy / truthy_raw) -------------------------- */

/* Control-flow truthiness with the UNRESOLVED discipline: an unprovable
 * condition is FALSE (skip). Only nil, false, and UNRESOLVED are falsy; 0 and
 * "" are TRUE. */
static inline int cv_truthy(CValue v) {
    if (cv_is_num(v)) return 1;              /* any number, incl 0.0, is truthy */
    unsigned t = cv_tag(v);
    return !(t == CV_TAG_NIL || t == CV_TAG_FALSE || t == CV_TAG_UNRESOLVED);
}

/* Raw truthiness for and/or operand selection - IGNORES the UNRESOLVED rule
 * (callers guard UNRESOLVED before reaching here). Only nil and false are
 * falsy. */
static inline int cv_truthy_raw(CValue v) {
    if (cv_is_num(v)) return 1;
    unsigned t = cv_tag(v);
    return !(t == CV_TAG_NIL || t == CV_TAG_FALSE);
}

#endif /* NOTITG_CVALUE_H */
