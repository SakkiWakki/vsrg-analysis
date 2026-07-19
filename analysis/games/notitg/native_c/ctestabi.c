/* A thin C ABI over the value core for ctypes differential-testing (Stage 0d).
 * NOT part of the executor - just a stable surface so a Python harness can fuzz
 * cv_binary/cv_unary/table ops against the frame_eval oracle. Values cross the
 * boundary as a tagged pair (tag:int, num:double, str:const char*) to avoid
 * leaking the NaN-box repr into ctypes. */
#include "cvalue.h"
#include "carena.h"
#include "cvalue_ops.h"
#include <string.h>

/* one shared arena for the test session */
static CArena *g_arena;
void   tabi_init(void)  { if (!g_arena) g_arena = carena_new(); }
void   tabi_reset(void) { if (g_arena) carena_free(g_arena); g_arena = carena_new(); }

/* External tag codes for the boundary (independent of the internal box tags). */
enum { X_NUM=0, X_NIL=1, X_TRUE=2, X_FALSE=3, X_UNRES=4, X_STR=5, X_TABLE=6 };

static CValue from_ext(int tag, double num, const char *str) {
    switch (tag) {
        case X_NUM:   return cv_num(num);
        case X_NIL:   return cv_nil();
        case X_TRUE:  return cv_bool(1);
        case X_FALSE: return cv_bool(0);
        case X_UNRES: return cv_unresolved();
        case X_STR:   return cv_str(carena_intern(g_arena, str, strlen(str)));
        default:      return cv_nil();
    }
}

/* Result crosses back as: return code = ext tag; out_num / out_str filled. */
static int to_ext(CValue v, double *out_num, const char **out_str) {
    if (cv_is_num(v))        { *out_num = cv_as_num(v); return X_NUM; }
    if (cv_is_nil(v))        return X_NIL;
    if (cv_is_unresolved(v)) return X_UNRES;
    if (cv_is_bool(v))       return cv_bool_val(v) ? X_TRUE : X_FALSE;
    if (cv_is_str(v))        { size_t l; *out_str = carena_str(g_arena, cv_payload(v), &l); return X_STR; }
    if (cv_is_table(v))      return X_TABLE;
    return X_NIL;
}

int tabi_binary(int op, int ta, double na, const char *sa,
                        int tb, double nb, const char *sb,
                        double *out_num, const char **out_str) {
    CValue r = cv_binary(g_arena, op, from_ext(ta, na, sa), from_ext(tb, nb, sb));
    return to_ext(r, out_num, out_str);
}

int tabi_unary(int op, int t, double n, const char *s,
               double *out_num, const char **out_str) {
    CValue r = cv_unary(g_arena, op, from_ext(t, n, s));
    return to_ext(r, out_num, out_str);
}
