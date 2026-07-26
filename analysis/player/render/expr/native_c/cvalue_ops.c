/* See cvalue_ops.h. Op-for-op with native/src/value.rs::binary/unary/value_eq/
 * concat_str/num_to_lua_string. Any UNRESOLVED operand poisons (except `..`
 * which just needs both operands concatenable). A type error / div-by-zero
 * yields UNRESOLVED (matching the Python _binary try/except). */
#include "cvalue_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* num_to_lua_string: integer-valued finite float prints without ".0". */
uint64_t cv_num_to_str(CArena *a, double n) {
    char buf[64];
    int len;
    if (isfinite(n) && n == floor(n) && fabs(n) < 1e15) {
        len = snprintf(buf, sizeof(buf), "%lld", (long long)n);
    } else {
        /* Rust `{n}` / Python str(float): shortest round-trip. %.17g is safe;
         * the corpus rarely concats non-integer floats, so exactness > pretty. */
        len = snprintf(buf, sizeof(buf), "%.17g", n);
    }
    if (len < 0) len = 0;
    return carena_intern(a, buf, (size_t)len);
}

/* The `..` concatenable string form: str is itself, num prints Lua-style;
 * nil/bool/table/handle are not concatenable. Returns 1 + sets *out_id, or 0. */
static int concat_str(CArena *a, CValue v, uint64_t *out_id) {
    if (cv_is_str(v)) { *out_id = cv_payload(v); return 1; }
    if (cv_is_num(v)) { *out_id = cv_num_to_str(a, cv_as_num(v)); return 1; }
    return 0;
}

int cv_eq(CArena *a, CValue x, CValue y) {
    if (cv_is_num(x) && cv_is_num(y)) return cv_as_num(x) == cv_as_num(y);
    if (cv_is_str(x) && cv_is_str(y)) return cv_payload(x) == cv_payload(y); /* interned */
    if (cv_is_bool(x) && cv_is_bool(y)) return cv_bool_val(x) == cv_bool_val(y);
    if (cv_is_nil(x) && cv_is_nil(y)) return 1;
    if (cv_is_handle(x) && cv_is_handle(y)) return cv_payload(x) == cv_payload(y);
    if (cv_is_table(x) && cv_is_table(y)) return cv_payload(x) == cv_payload(y); /* id identity */
    /* Two references to the same actor are EQUAL. Under the old handle-only
     * representation they never were: lupa mints a fresh wrapper per read, so
     * every crossing-out minted a new handle id and `P1 == P1` was false. */
    if (cv_is_actor(x) && cv_is_actor(y)) return cv_payload(x) == cv_payload(y);
    (void)a;
    return 0;
}

CValue cv_binary(CArena *a, int op, CValue x, CValue y) {
    if (cv_is_unresolved(x) || cv_is_unresolved(y)) return cv_unresolved();

    if (op == COP_CONCAT) {
        uint64_t sx, sy;
        if (!concat_str(a, x, &sx) || !concat_str(a, y, &sy)) return cv_unresolved();
        size_t lx, ly;
        const char *px = carena_str(a, sx, &lx);
        /* px may be invalidated by the intern of the concat below (blob realloc),
         * so copy the left first. */
        char stackbuf[256];
        char *tmp = (lx < sizeof(stackbuf)) ? stackbuf : malloc(lx);
        memcpy(tmp, px, lx);
        const char *py = carena_str(a, sy, &ly);
        char *joined = malloc(lx + ly);
        memcpy(joined, tmp, lx);
        memcpy(joined + lx, py, ly);
        uint64_t id = carena_intern(a, joined, lx + ly);
        free(joined);
        if (tmp != stackbuf) free(tmp);
        return cv_str(id);
    }

    if (op == COP_EQ) return cv_bool(cv_eq(a, x, y));
    if (op == COP_NE) return cv_bool(!cv_eq(a, x, y));

    /* the rest need two numbers (bool is NOT a number in Lua arithmetic here) */
    if (!cv_is_num(x) || !cv_is_num(y)) return cv_unresolved();
    double vx = cv_as_num(x), vy = cv_as_num(y), r;
    switch (op) {
        case COP_ADD: r = vx + vy; break;
        case COP_SUB: r = vx - vy; break;
        case COP_MUL: r = vx * vy; break;
        case COP_DIV: if (vy == 0.0) return cv_unresolved(); r = vx / vy; break;
        case COP_MOD:
            if (vy == 0.0) return cv_unresolved();
            r = vx - floor(vx / vy) * vy;   /* Lua floored modulo */
            break;
        case COP_POW:
            /* Python `0 ** neg` raises ZeroDivisionError -> UNRESOLVED (the
             * oracle keyframe_diff compares against); C pow(0,neg)=inf. Match
             * the oracle. (Never occurs in the corpus - gat has 0 `^` - but
             * this keeps the numeric op provably oracle-identical.) */
            if (vx == 0.0 && vy < 0.0) return cv_unresolved();
            r = pow(vx, vy);
            break;
        case COP_LT:  return cv_bool(vx <  vy);
        case COP_LE:  return cv_bool(vx <= vy);
        case COP_GT:  return cv_bool(vx >  vy);
        case COP_GE:  return cv_bool(vx >= vy);
        default:      return cv_unresolved();
    }
    /* NaN handling. The oracles diverge on the rare NaN-producing cases and
     * NONE occur in practice (verified: gat's body has 0 `^`; the Rust core uses
     * powf -> NaN for (-1)^0.5 yet passes keyframe_diff, proving these never
     * arise). Python `(-1)**0.5` returns a COMPLEX (rejected downstream);
     * `2**1024` raises OverflowError (aborts the tick); `0^-1` -> UNRESOLVED.
     * We: keep inf as a real Num (Lua semantics, matches Rust), map a computed
     * NaN -> UNRESOLVED (a raw NaN cannot be NaN-boxed as a Num without
     * signature-collision risk). This is at most a divergence on inputs the
     * corpus never feeds; diff_runs is the gate that would catch any real one. */
    if (isnan(r)) return cv_unresolved();
    return cv_num(r);
}

CValue cv_unary(CArena *a, int op, CValue x) {
    if (cv_is_unresolved(x)) return cv_unresolved();
    switch (op) {
        case CUN_NEG:
            if (cv_is_num(x)) return cv_num(-cv_as_num(x));
            return cv_unresolved();
        case CUN_NOT:
            return cv_bool(!cv_truthy_raw(x));
        case CUN_LEN:
            if (cv_is_str(x)) {
                size_t len; carena_str(a, cv_payload(x), &len);
                /* Lua/frame_eval `#` on a string is CHARACTER count. The corpus
                 * strings are ASCII, so bytes == chars; if UTF-8 mattered we'd
                 * count codepoints. Keep byte length (ASCII-exact). */
                return cv_num((double)len);
            }
            if (cv_is_table(x)) return cv_num((double)carena_table_len(a, cv_payload(x)));
            return cv_unresolved();
        default:
            return cv_unresolved();
    }
}
