/* Value operations - binary / unary / concat / equality - matching
 * native/src/value.rs op-for-op (which mirrors the Python frame_eval `_binary`).
 * These need the arena for string concat / equality / length, so they take a
 * CArena*. Binary op ids are small ints the compiler resolves once (no per-tick
 * string dispatch). */
#ifndef NOTITG_CVALUE_OPS_H
#define NOTITG_CVALUE_OPS_H

#include "cvalue.h"
#include "carena.h"

/* Binary op ids (compiler-resolved from the AST op string). */
enum {
    COP_ADD, COP_SUB, COP_MUL, COP_DIV, COP_MOD, COP_POW,
    COP_LT, COP_LE, COP_GT, COP_GE, COP_EQ, COP_NE, COP_CONCAT,
};

/* Unary op ids. */
enum { CUN_NEG, CUN_NOT, CUN_LEN };

CValue cv_binary(CArena *a, int op, CValue x, CValue y);
CValue cv_unary(CArena *a, int op, CValue x);

/* Lua number -> string (integer-valued float drops the .0). Returns an interned
 * string id via the arena. Used by `..` and tostring. */
uint64_t cv_num_to_str(CArena *a, double n);

/* Value equality (Lua ==): num/str/bool/nil by value, handle/table/func by
 * identity (id equality). */
int cv_eq(CArena *a, CValue x, CValue y);

#endif /* NOTITG_CVALUE_OPS_H */
