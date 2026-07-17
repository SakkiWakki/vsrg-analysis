"""StepMania bitmap-font parsing for storyboard 'bitmaptext' elements.

A StepMania font is an `.ini` (glyph metrics) beside a texture PNG whose
filename encodes the glyph grid: `<name> [COLS]x[ROWS].png` (e.g.
`_eurostile normal (mipmaps) 16x16.png` = a 16x16 grid of 32x32 cells).
Glyph cell index equals the character codepoint (the default identity
map; explicit `[map ...]` sections are not modelled - the pilot's fonts
use the identity layout), so char C lives at grid cell C: column
`C % cols`, row `C // cols`.

The `.ini`'s `[Char Widths]` gives the ADVANCE width per codepoint (how
far the pen moves), `LineSpacing`/`Baseline`/`Top` the vertical metrics,
`AddToAllWidths`/`AdvanceExtraPixels` global tweaks. The glyph itself is
the whole cell, drawn centred on the pen so ink lines up regardless of
the (usually wider) cell; only the advance uses the metric width. That
faithfully reproduces StepMania's own centred-glyph drawing without
needing per-glyph ink bounds.

This module resolves a `File=` reference to its .ini + texture, parses
the metrics, and exposes glyph source rects + advances so the renderer
can composite a string from the atlas. Fonts load once and cache; a
reference that cannot be resolved returns None so the caller falls back
to a system font.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_GRID_RE = re.compile(r'(?P<cols>\d+)x(?P<rows>\d+)(?=\D*$)')
_SECTION_RE = re.compile(r'^\s*\[(?P<name>[^\]]+)\]\s*$')
_KV_RE = re.compile(r'^\s*(?P<key>[^=;#]+?)\s*=\s*(?P<value>[^;/]*)')


@dataclass(frozen=True)
class BitmapFont:
    """A parsed SM font: one texture grid plus per-codepoint advances.
    Cell pixel size is derived from the atlas dimensions at draw time
    (the renderer owns the pixmap), so this record stays pixmap-free."""
    texture_path: str
    cols: int
    rows: int
    line_spacing: float
    default_advance: float
    advances: dict          # codepoint -> advance width, design px

    def advance(self, codepoint: int) -> float:
        return self.advances.get(codepoint, self.default_advance)

    def cell(self, codepoint: int, atlas_w: float, atlas_h: float):
        """(x, y, w, h) source rect of this codepoint's cell in an atlas
        of the given pixel size, or None when the codepoint is outside
        the grid."""
        count = self.cols * self.rows
        if not 0 <= codepoint < count:
            return None
        cell_w = atlas_w / self.cols
        cell_h = atlas_h / self.rows
        col = codepoint % self.cols
        row = codepoint // self.cols
        return (col * cell_w, row * cell_h, cell_w, cell_h)


def _resolve_ini(reference: str, search_dirs) -> Path | None:
    """Find the `.ini` for a `File=` font reference. A reference is a
    base name (`_eurostile normal`) resolved against the lua dir and the
    theme Fonts dirs; an explicit path with a real .ini wins outright."""
    direct = Path(reference)
    if direct.suffix.lower() == '.ini' and direct.is_file():
        return direct
    for base in search_dirs:
        candidate = Path(base) / f'{reference}.ini'
        if candidate.is_file():
            return candidate
    return None


def _find_texture(ini_path: Path) -> tuple | None:
    """The grid texture beside `ini_path`: a `<stem>*[COLS]x[ROWS].*`
    image. Returns (path, cols, rows) or None. Prefers a non-mipmap page
    when several match, else the first."""
    stem = ini_path.stem
    matches = []
    for sibling in ini_path.parent.iterdir():
        if sibling.stem.startswith(stem) and sibling.suffix.lower() in (
                '.png', '.jpg', '.jpeg', '.bmp', '.gif'):
            grid = _GRID_RE.search(sibling.stem)
            if grid:
                matches.append((sibling, int(grid['cols']), int(grid['rows'])))
    if not matches:
        return None
    matches.sort(key=lambda m: ('mipmap' in m[0].stem.lower(), m[0].name))
    return matches[0]


def _parse_ini(text: str) -> dict:
    """`[Char Widths]` metrics: {'widths': {codepoint: px}, plus the
    scalar keys}. Only the `[Char Widths]` section is read; the pilot's
    fonts keep everything there."""
    widths = {}
    scalars = {}
    in_widths = False
    for line in text.splitlines():
        section = _SECTION_RE.match(line)
        if section:
            in_widths = section['name'].strip().lower() == 'char widths'
            continue
        if not in_widths:
            continue
        kv = _KV_RE.match(line)
        if not kv:
            continue
        key, value = kv['key'].strip(), kv['value'].strip()
        if key.isdigit():
            width = _as_int(value)
            if width is not None:
                widths[int(key)] = width
        else:
            scalars[key.lower()] = value
    return {'widths': widths, 'scalars': scalars}


def load_font(reference: str, search_dirs) -> BitmapFont | None:
    """Parse the SM font named by a `File=` reference, or None when it
    cannot be resolved. `search_dirs` are directories to probe for
    `<reference>.ini` (the chart's lua dir, the theme Fonts dirs)."""
    ini_path = _resolve_ini(reference, search_dirs)
    if ini_path is None:
        return None
    texture = _find_texture(ini_path)
    if texture is None:
        return None

    texture_path, cols, rows = texture
    parsed = _parse_ini(ini_path.read_text(encoding='utf-8', errors='replace'))
    scalars = parsed['scalars']
    extra = _as_int(scalars.get('advanceextrapixels'), 0)
    add_all = _as_int(scalars.get('addtoallwidths'), 0)
    advances = {cp: w + extra + add_all for cp, w in parsed['widths'].items()}
    default_advance = float(max(advances.values(), default=16))
    return BitmapFont(
        texture_path=str(texture_path), cols=cols, rows=rows,
        line_spacing=float(_as_int(scalars.get('linespacing'), 0) or 0),
        default_advance=default_advance, advances=advances)


def _as_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
