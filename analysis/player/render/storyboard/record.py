"""The DrawSchedule record layout: the one Python-side statement of the op
stream `storyboard_native` emits.

The AUTHORITY is `native/src/evaluate.rs`; this is the single Python mirror of
it, and `test_record_layout.py` asserts the strides against a live evaluator so
the mirror cannot drift silently. Everything that reads a record - the GL
executor, the raster executor, the doc compiler's parity harnesses, tests
hand-building rows - imports from here rather than restating offsets.

That rule exists because restating them has failed twice: a test carrying its
own ``_F_STRIDE = 20`` indexed past the end of its own row the moment a lane
was added, and the async prepare path mirrored the builder API by hand and
lost a method. A lane added in Rust needs exactly two edits: here, and the
executor that consumes it.

Deliberately Qt-free and extension-free: the doc compiler imports this on a
worker thread, and the pure helpers around it must stay importable without
either.
"""
from __future__ import annotations

# Record strides. Checked against `evaluator.u_stride` / `f_stride`.
U_STRIDE = 11
F_STRIDE = 28

# u32 lanes.
U_KIND = 0
U_A = 1
U_B = 2
U_C = 3        # BLIT src_aux / sheet frame
U_BLEND = 4
U_SHADER = 5   # shader id + 1; 0 = unshaded
U_CLIP = 6     # clip id + 1; 0 = unclipped
U_SCREEN_SPACE = 7
U_UF_OFFSET = 8
U_UF_COUNT = 9
U_TAG = 10     # caller-assigned item id; 0 = untagged, diagnostics only

# f32 lanes.
F_MAT = 0       # ..9  (mat3, row-major, column-vector convention)
F_OPACITY = 9
F_TINT = 10     # ..13 (rgb)
F_CROP = 13     # ..17 (l, t, r, b as fractions of the SOURCE logical box)
F_ORIGIN = 17   # ..19 (x, y as fractions of the item's own drawn size)
F_SIZE = 19     # ..21 (absolute w, h replacing the natural box; < 0 = natural)
F_FIT = 21      # ..24 (ScaleToCover/FitInside: mode, rect w, rect h)
F_FADE = 24     # ..28 (SetFade l, r, t, b as fractions of the drawn box)

# Op kinds.
OP_BEGIN = 0
OP_BLIT = 1
OP_COPY = 2
OP_END = 3

# Source kinds. Field instances only ever blit fills and drawables, so an
# IMAGE blit in a NotITG doc is unambiguously a storyboard element.
SRC_IMAGE = 0
SRC_DRAWABLE = 1
SRC_MESH = 2
SRC_FILL = 3
SRC_LINES = 4

# Clear modes (ClearMode in doc.rs).
CLEAR_TRANSPARENT = 0
CLEAR_OPAQUE = 1
CLEAR_RETAIN = 2

# Blend modes (Blend in doc.rs).
BLEND_SOURCE_OVER = 0
BLEND_ADDITIVE = 1

# Sentinels a lane uses to mean "unset", rather than a magic number at each
# reader. A negative size keeps the source's natural box; a fit mode below
# this is off; ScaleToCover takes the LARGER axis ratio, anything else the
# smaller.
SIZE_NATURAL = -1.0
FIT_OFF_BELOW = 0.5
FIT_COVER = 1.0
