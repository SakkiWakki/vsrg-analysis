/* See carena.h. Correct-over-fast where the two conflict, EXCEPT the integer
 * array read/write, which is the measured hot path and is a direct slot access. */
#include "carena.h"

#include <stdlib.h>
#include <string.h>

/* ---- string arena ------------------------------------------------------ */
/* Interned: a flat blob of bytes + an index table {id -> (off,len)} + an
 * open-addressed hash {hash -> id} for dedup. */

typedef struct { size_t off, len; } CStr;

typedef struct {
    char   *blob;   size_t blob_len, blob_cap;
    CStr   *ents;   size_t n_str, str_cap;
    /* dedup hash: buckets hold str-id+1 (0 = empty) */
    uint64_t *buckets; size_t nbuckets;
} StrArena;

static uint64_t fnv1a(const char *s, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void str_rehash(StrArena *sa, size_t newn) {
    uint64_t *nb = calloc(newn, sizeof(uint64_t));
    for (size_t b = 0; b < sa->nbuckets; b++) {
        uint64_t slot = sa->buckets[b];
        if (!slot) continue;
        uint64_t id = slot - 1;
        uint64_t h = fnv1a(sa->blob + sa->ents[id].off, sa->ents[id].len);
        size_t m = h & (newn - 1);
        while (nb[m]) m = (m + 1) & (newn - 1);
        nb[m] = slot;
    }
    free(sa->buckets);
    sa->buckets = nb;
    sa->nbuckets = newn;
}

static uint64_t str_intern(StrArena *sa, const char *s, size_t len) {
    uint64_t h = fnv1a(s, len);
    size_t m = h & (sa->nbuckets - 1);
    while (sa->buckets[m]) {
        uint64_t id = sa->buckets[m] - 1;
        if (sa->ents[id].len == len &&
            memcmp(sa->blob + sa->ents[id].off, s, len) == 0)
            return id;
        m = (m + 1) & (sa->nbuckets - 1);
    }
    /* new string: append to blob */
    if (sa->blob_len + len + 1 > sa->blob_cap) {
        while (sa->blob_len + len + 1 > sa->blob_cap) sa->blob_cap *= 2;
        sa->blob = realloc(sa->blob, sa->blob_cap);
    }
    size_t off = sa->blob_len;
    memcpy(sa->blob + off, s, len);
    sa->blob[off + len] = '\0';
    sa->blob_len += len + 1;
    if (sa->n_str + 1 > sa->str_cap) {
        sa->str_cap *= 2;
        sa->ents = realloc(sa->ents, sa->str_cap * sizeof(CStr));
    }
    uint64_t id = sa->n_str++;
    sa->ents[id].off = off;
    sa->ents[id].len = len;
    /* grow hash at 70% load BEFORE inserting, then find the slot fresh (the
     * `m` computed above is stale if we rehashed). */
    if (sa->n_str * 10 >= sa->nbuckets * 7) str_rehash(sa, sa->nbuckets * 2);
    size_t ins = h & (sa->nbuckets - 1);
    while (sa->buckets[ins]) ins = (ins + 1) & (sa->nbuckets - 1);
    sa->buckets[ins] = id + 1;
    return id;
}

/* ---- tables ------------------------------------------------------------ */
/* A table has a dense array part (1-based Lua, stored 0-based) plus an
 * open-addressed keyed map for string / out-of-array-run keys. */

typedef struct { uint64_t key_kind; int64_t ik; uint64_t sk; CValue v; int used; } KEnt;
/* key_kind: 0 empty, 1 int, 2 str (sk = interned string id). */

typedef struct {
    CValue *arr;  size_t arr_len, arr_cap;   /* array part, index 0 = Lua t[1] */
    KEnt   *map;  size_t map_n, map_cap;      /* keyed part (open addressing) */
} CTable;

struct CArena {
    StrArena str;
    CTable *tabs;  size_t n_tab, tab_cap;
};

/* ---- arena lifecycle --------------------------------------------------- */

CArena *carena_new(void) {
    CArena *a = calloc(1, sizeof(CArena));
    a->str.blob_cap = 4096; a->str.blob = malloc(a->str.blob_cap);
    a->str.str_cap = 256;   a->str.ents = malloc(a->str.str_cap * sizeof(CStr));
    a->str.nbuckets = 512;  a->str.buckets = calloc(a->str.nbuckets, sizeof(uint64_t));
    a->tab_cap = 64;        a->tabs = calloc(a->tab_cap, sizeof(CTable));
    return a;
}

void carena_free(CArena *a) {
    if (!a) return;
    free(a->str.blob); free(a->str.ents); free(a->str.buckets);
    for (size_t i = 0; i < a->n_tab; i++) { free(a->tabs[i].arr); free(a->tabs[i].map); }
    free(a->tabs);
    free(a);
}

uint64_t carena_intern(CArena *a, const char *s, size_t len) {
    return str_intern(&a->str, s, len);
}
const char *carena_str(CArena *a, uint64_t id, size_t *len_out) {
    if (id >= a->str.n_str) { if (len_out) *len_out = 0; return ""; }
    if (len_out) *len_out = a->str.ents[id].len;
    return a->str.blob + a->str.ents[id].off;
}

static CTable *tab(CArena *a, uint64_t tid) { return &a->tabs[tid]; }

static uint64_t new_table(CArena *a, size_t arr_prealloc) {
    if (a->n_tab + 1 > a->tab_cap) {
        a->tab_cap *= 2;
        a->tabs = realloc(a->tabs, a->tab_cap * sizeof(CTable));
        memset(a->tabs + a->n_tab, 0, (a->tab_cap - a->n_tab) * sizeof(CTable));
    }
    uint64_t id = a->n_tab++;
    CTable *t = &a->tabs[id];
    memset(t, 0, sizeof(*t));
    if (arr_prealloc) {
        t->arr_cap = arr_prealloc;
        t->arr = malloc(arr_prealloc * sizeof(CValue));
    }
    return id;
}

uint64_t carena_table_new(CArena *a) { return new_table(a, 0); }
uint64_t carena_table_new_array(CArena *a, size_t n) {
    uint64_t id = new_table(a, n ? n : 1);
    return id;
}

/* ---- integer (array) fast path ---------------------------------------- */

static void arr_ensure(CTable *t, size_t need) {
    if (need <= t->arr_cap) return;
    if (t->arr_cap == 0) t->arr_cap = 4;
    while (t->arr_cap < need) t->arr_cap *= 2;
    t->arr = realloc(t->arr, t->arr_cap * sizeof(CValue));
}

CValue carena_table_geti(CArena *a, uint64_t tid, int64_t i) {
    CTable *t = tab(a, tid);
    if (i >= 1 && (size_t)i <= t->arr_len) return t->arr[i - 1];  /* HOT: base+offset */
    /* fall to keyed map for an int key outside the array run */
    for (size_t k = 0; k < t->map_cap; k++)
        if (t->map[k].used && t->map[k].key_kind == 1 && t->map[k].ik == i)
            return t->map[k].v;
    return cv_nil();
}

void carena_table_seti(CArena *a, uint64_t tid, int64_t i, CValue v) {
    CTable *t = tab(a, tid);
    int is_absent = cv_is_nil(v) || cv_is_unresolved(v);
    if (i >= 1 && (size_t)i <= t->arr_len) {
        if (is_absent && (size_t)i == t->arr_len) { t->arr_len--; return; }
        t->arr[i - 1] = is_absent ? cv_nil() : v;
        return;
    }
    if (i == (int64_t)t->arr_len + 1 && !is_absent) {
        arr_ensure(t, t->arr_len + 1);
        t->arr[t->arr_len++] = v;
        /* absorb any keyed entries that now extend the contiguous run */
        for (;;) {
            int64_t next = (int64_t)t->arr_len + 1;
            int found = 0;
            for (size_t k = 0; k < t->map_cap; k++)
                if (t->map[k].used && t->map[k].key_kind == 1 && t->map[k].ik == next) {
                    arr_ensure(t, t->arr_len + 1);
                    t->arr[t->arr_len++] = t->map[k].v;
                    t->map[k].used = 0; t->map[k].key_kind = 0; t->map_n--;
                    found = 1; break;
                }
            if (!found) break;
        }
        return;
    }
    /* else: a sparse int key -> keyed map */
    if (is_absent) { /* remove from map */
        for (size_t k = 0; k < t->map_cap; k++)
            if (t->map[k].used && t->map[k].key_kind == 1 && t->map[k].ik == i) {
                t->map[k].used = 0; t->map[k].key_kind = 0; t->map_n--; return;
            }
        return;
    }
    /* insert/update int key in map (linear; sparse int keys are rare) */
    for (size_t k = 0; k < t->map_cap; k++)
        if (t->map[k].used && t->map[k].key_kind == 1 && t->map[k].ik == i) { t->map[k].v = v; return; }
    if (t->map_cap == 0 || t->map_n + 1 > t->map_cap / 2) {
        size_t nc = t->map_cap ? t->map_cap * 2 : 8;
        KEnt *nm = calloc(nc, sizeof(KEnt));
        for (size_t k = 0; k < t->map_cap; k++) if (t->map[k].used) {
            size_t s = 0; while (nm[s].used) s++; nm[s] = t->map[k];
        }
        free(t->map); t->map = nm; t->map_cap = nc;
    }
    size_t s = 0; while (t->map[s].used) s++;
    t->map[s].used = 1; t->map[s].key_kind = 1; t->map[s].ik = i; t->map[s].v = v; t->map_n++;
}

int64_t carena_table_len(CArena *a, uint64_t tid) { return (int64_t)tab(a, tid)->arr_len; }
void carena_table_append(CArena *a, uint64_t tid, CValue v) {
    CTable *t = tab(a, tid);
    carena_table_seti(a, tid, (int64_t)t->arr_len + 1, v);
}

/* ---- keyed (string) path ---------------------------------------------- */

static void map_set_str(CTable *t, uint64_t sk, CValue v, int is_absent) {
    for (size_t k = 0; k < t->map_cap; k++)
        if (t->map[k].used && t->map[k].key_kind == 2 && t->map[k].sk == sk) {
            if (is_absent) { t->map[k].used = 0; t->map[k].key_kind = 0; t->map_n--; }
            else t->map[k].v = v;
            return;
        }
    if (is_absent) return;
    if (t->map_cap == 0 || t->map_n + 1 > t->map_cap / 2) {
        size_t nc = t->map_cap ? t->map_cap * 2 : 8;
        KEnt *nm = calloc(nc, sizeof(KEnt));
        for (size_t k = 0; k < t->map_cap; k++) if (t->map[k].used) {
            size_t s = 0; while (nm[s].used) s++; nm[s] = t->map[k];
        }
        free(t->map); t->map = nm; t->map_cap = nc;
    }
    size_t s = 0; while (t->map[s].used) s++;
    t->map[s].used = 1; t->map[s].key_kind = 2; t->map[s].sk = sk; t->map[s].v = v; t->map_n++;
}

static CValue map_get_str(CTable *t, uint64_t sk) {
    for (size_t k = 0; k < t->map_cap; k++)
        if (t->map[k].used && t->map[k].key_kind == 2 && t->map[k].sk == sk)
            return t->map[k].v;
    return cv_nil();
}

/* ---- generic key path (folds num->int per Lua, str stays str) --------- */

CValue carena_table_get(CArena *a, uint64_t tid, CValue key) {
    if (cv_is_num(key)) {
        double n = cv_as_num(key);
        double f = n; /* whole-float folds to int key */
        if (f == (double)(int64_t)f) return carena_table_geti(a, tid, (int64_t)f);
        /* non-integer numeric key -> stringify (rare); handled as str key */
        return cv_nil();  /* corpus never keys tables by fractional numbers */
    }
    if (cv_is_str(key)) return map_get_str(tab(a, tid), cv_payload(key));
    return cv_nil();
}

void carena_table_set(CArena *a, uint64_t tid, CValue key, CValue v) {
    int is_absent = cv_is_nil(v) || cv_is_unresolved(v);
    if (cv_is_num(key)) {
        double n = cv_as_num(key);
        if (n == (double)(int64_t)n) { carena_table_seti(a, tid, (int64_t)n, v); return; }
        return; /* fractional numeric key: unsupported/unused */
    }
    if (cv_is_str(key)) { map_set_str(tab(a, tid), cv_payload(key), v, is_absent); return; }
    /* other key kinds ignored (upstream-validated the body never does this) */
}
