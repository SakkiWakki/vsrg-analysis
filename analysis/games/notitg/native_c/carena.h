/* Kernel-owned string + table arenas for the NotITG residue executor.
 *
 * The hot path (measured: 2.1M/150-tick) is INTEGER indexing into snapshotted
 * read-only DATA tables that upstream deep-copied from lupa host tables (pure
 * nested arrays - `native_frontier._deep_copy`). So the table model is built for
 * that case: a dense `CValue[]` array part with O(1) `t[i]` (base+offset load,
 * no hashing, no refcount, no borrow-check), plus a keyed map fallback for the
 * rare string-keyed / sparse table.
 *
 * Semantics mirror `native/src/table.rs` + Python `frame_eval.LuaTable`:
 *   - Lua has no int/float distinction: a whole-valued float key folds to the
 *     integer key (keeps array runs contiguous). A non-integer numeric key
 *     stringifies.
 *   - Storing nil/UNRESOLVED REMOVES the key (absent == nil, length stays
 *     contiguous).
 *   - `t[absent]` reads back NIL (a resolved value), not UNRESOLVED.
 *   - `#t` is the border of the 1..n contiguous array run.
 *
 * Lifetime: arenas are owned by a `CArena` created per body-compile and freed on
 * chart reload. Snapshotted tables are immutable after build (no per-tick
 * allocation on the hot path). Ids are 48-bit indices NaN-boxed into a CValue.
 */
#ifndef NOTITG_CARENA_H
#define NOTITG_CARENA_H

#include <stdint.h>
#include "cvalue.h"

typedef struct CArena CArena;

CArena *carena_new(void);
void    carena_free(CArena *a);

/* --- strings (interned) ------------------------------------------------- */
/* Intern `s` (len bytes, not necessarily nul-terminated); returns the string
 * id. Equal byte sequences return the same id (so string == is an id compare
 * on the fast path, matching Lua interned-string equality). */
uint64_t carena_intern(CArena *a, const char *s, size_t len);
/* Resolve an id back to bytes (borrowed, arena-owned, valid until free). */
const char *carena_str(CArena *a, uint64_t id, size_t *len_out);

/* --- tables ------------------------------------------------------------- */
/* Create an empty table; returns its id. */
uint64_t carena_table_new(CArena *a);
/* Create a dense array table pre-sized for `n` elements (the snapshot fast
 * path); elements start nil, fill with carena_table_seti(0-based via 1-based
 * Lua keys). */
uint64_t carena_table_new_array(CArena *a, size_t n);

/* t[key] = v  (key is a CValue: num->folded int/str, str->str key). A
 * nil/UNRESOLVED v removes the key. Non-num/str keys are ignored (Lua would
 * error; the body never does this - upstream-validated). */
void   carena_table_set(CArena *a, uint64_t tid, CValue key, CValue v);
/* t[key] read; absent -> cv_nil(). */
CValue carena_table_get(CArena *a, uint64_t tid, CValue key);

/* Fast integer path for the hot loop: t[i] with a 1-based Lua index, no key
 * boxing. Out-of-range / hole -> cv_nil(). */
CValue carena_table_geti(CArena *a, uint64_t tid, int64_t i);
void   carena_table_seti(CArena *a, uint64_t tid, int64_t i, CValue v);

/* #t : the contiguous 1..n array border. */
int64_t carena_table_len(CArena *a, uint64_t tid);
/* table.insert(t, v): append at the array border. */
void    carena_table_append(CArena *a, uint64_t tid, CValue v);

#endif /* NOTITG_CARENA_H */
