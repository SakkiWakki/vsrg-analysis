"""fluXis `.fsb` storyboard -> game-agnostic storyboard IR.

The `.fsb` is Newtonsoft-serialized `fluXis.Storyboards.Storyboard`:
a design `resolution` plus `elements`, each with typed keyframe
`animations`. Compilation notes (all verified against fluXis source):

- Version 1 files (missing/1 `version`) carry ABSOLUTE animation start
  times; version 2 made them relative to the element's start (the
  in-game migration subtracts element start). The IR wants absolute
  seconds, so v2 adds the element start back.
- `use-start` on an animation defaults to TRUE (Json ignores the
  default): the property snaps to `start-value` at the animation's
  start; without it the value eases from wherever it currently is,
  which is EventTimeline's default previous-keyframe behavior.
- Sprites auto-size to their texture (DrawableStoryboardSprite sets
  AutoSizeAxes); the element's width/height JSON is ignored for them.
- Element `color` is packed 0xRRGGBBAA (Colour4.FromRGBA); its alpha
  folds into the alpha timeline's rest state. Color animation values
  are hex strings.
- Anchor/origin are osu-framework `Anchor` bitmasks.
- Scale (scalar) and ScaleVector both drive the same underlying scale,
  so both feed the scale_x/scale_y timelines chronologically.

Unsupported element types are skipped with one warning each: Script
(Lua), SkinSprite (needs our skin), Compound, Video.

Layer z-slots mirror fluXis's gameplay nesting: Background sits above
the map background but below the field; Foreground above the field;
Overlay above that (our sidebar HUD stays on top by design).
"""
from __future__ import annotations

import json
from pathlib import Path

from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.storyboard import Element, Storyboard
from analysis.player.render.storyboard.model import build_timelines

_LAYER_Z = {0: -900, 1: 400, 2: 700}

_KINDS = {0: 'rect', 1: 'sprite', 2: 'text', 4: 'ellipse',
          5: 'outline_ellipse', 7: 'outline_rect'}
_UNSUPPORTED = {3: 'Script', 6: 'SkinSprite', 8: 'Compound', 9: 'Video'}

_ANCHOR_X = ((32, 1.0), (16, 0.5), (8, 0.0))
_ANCHOR_Y = ((4, 1.0), (2, 0.5), (1, 0.0))

_SCALAR_PROPS = {0: 'x', 1: 'y', 4: 'w', 5: 'h', 6: 'rotation',
                 7: 'alpha', 9: 'border'}
_ANIM_SCALE, _ANIM_SCALE_VECTOR, _ANIM_COLOR = 2, 3, 8

_DEFAULT_BORDER = 4.0


def _anchor_frac(bits: int) -> tuple:
    x = next((f for bit, f in _ANCHOR_X if bits & bit), 0.0)
    y = next((f for bit, f in _ANCHOR_Y if bits & bit), 0.0)
    return (x, y)


def _rgba(packed: int) -> tuple:
    return ((packed >> 24 & 0xFF) / 255.0, (packed >> 16 & 0xFF) / 255.0,
            (packed >> 8 & 0xFF) / 255.0, (packed & 0xFF) / 255.0)


def _hex_rgb(value: str) -> tuple:
    text = str(value).lstrip('#')
    try:
        packed = int(text[:6], 16)
    except ValueError:
        return (1.0, 1.0, 1.0)
    return ((packed >> 16 & 0xFF) / 255.0, (packed >> 8 & 0xFF) / 255.0,
            (packed & 0xFF) / 255.0)


def _floats(value, arity: int) -> tuple:
    parts = str(value).split(',')
    try:
        floats = tuple(float(p) for p in parts[:arity])
    except ValueError:
        return (0.0,) * arity
    return floats + (0.0,) * (arity - len(floats))


def _anim_targets(anim) -> tuple:
    """((prop, start_values, end_values), ...) for one animation; empty
    when the type is unknown."""
    kind = int(anim.get('type', -1))
    start_raw = anim.get('start-value', '0')
    end_raw = anim.get('end-value', '0')

    if kind in _SCALAR_PROPS:
        prop = _SCALAR_PROPS[kind]
        return ((prop, _floats(start_raw, 1), _floats(end_raw, 1)),)
    if kind == _ANIM_SCALE:
        start, end = _floats(start_raw, 1), _floats(end_raw, 1)
        return (('scale_x', start, end), ('scale_y', start, end))
    if kind == _ANIM_SCALE_VECTOR:
        sx, sy = _floats(start_raw, 2)
        ex, ey = _floats(end_raw, 2)
        return (('scale_x', (sx,), (ex,)), ('scale_y', (sy,), (ey,)))
    if kind == _ANIM_COLOR:
        return (('color', _hex_rgb(start_raw), _hex_rgb(end_raw)),)
    return ()


def _element_keyframes(el, version: int) -> dict:
    el_start_ms = float(el.get('start', 0.0))
    keyframes: dict = {}
    for anim in el.get('animations') or []:
        if not isinstance(anim, dict):
            continue
        start_ms = float(anim.get('start', 0.0))
        if version >= 2:
            start_ms += el_start_ms
        duration_s = max(0.0, float(anim.get('duration', 0.0) or 0.0)) / 1000.0
        easing = int(anim.get('easing', 0) or 0)
        use_start = bool(anim.get('use-start', True))
        for prop, start_values, end_values in _anim_targets(anim):
            keyframes.setdefault(prop, []).append(Keyframe(
                t=start_ms / 1000.0,
                values=end_values,
                duration=duration_s,
                easing=easing,
                start=start_values if use_start else None,
            ))
    return keyframes


def _compile_element(el, version, assets_dir, warned) -> Element | None:
    raw_kind = int(el.get('type', -1))
    kind = _KINDS.get(raw_kind)
    if kind is None:
        name = _UNSUPPORTED.get(raw_kind, f'type {raw_kind}')
        if name not in warned:
            warned.add(name)
            print(f'fluXis storyboard: {name} elements not supported yet;'
                  f' skipping')
        return None

    params = el.get('parameters') or {}
    asset = None
    if kind == 'sprite':
        file_name = str(params.get('file', '')).strip()
        if not file_name:
            return None
        asset = str(assets_dir / file_name)

    r, g, b, color_a = _rgba(int(el.get('color', 0xFFFFFFFF)))
    rests = {
        'x': float(el.get('x', 0.0)),
        'y': float(el.get('y', 0.0)),
        'w': float(el.get('width', 0.0)),
        'h': float(el.get('height', 0.0)),
        'alpha': color_a,
        'border': float(params.get('border', _DEFAULT_BORDER)),
        'color': (r, g, b),
    }

    return Element(
        kind=kind,
        z=_LAYER_Z.get(int(el.get('layer', 0)), _LAYER_Z[0]),
        z_index=int(el.get('z-index', 0)),
        t_start=float(el.get('start', 0.0)) / 1000.0,
        t_end=float(el.get('end', 0.0)) / 1000.0,
        anchor=_anchor_frac(int(el.get('anchor', 0))),
        origin=_anchor_frac(int(el.get('origin', 0))),
        timelines=build_timelines(rests, _element_keyframes(el, version)),
        asset=asset,
        text=str(params.get('text', '')),
        font_px=float(params.get('size', 0.0) or 0.0),
        additive=bool(el.get('blend', False)),
    )


def parse_fsb(fsb_path) -> Storyboard | None:
    """Compile a `.fsb` into the storyboard IR; None when the file is
    absent/unreadable or has no supported elements."""
    fsb_path = Path(fsb_path)
    try:
        with open(fsb_path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    version = int(raw.get('version') or 1)
    resolution = raw.get('resolution') or {}
    warned: set = set()
    elements = [
        compiled for el in raw.get('elements') or [] if isinstance(el, dict)
        if (compiled := _compile_element(el, version, fsb_path.parent,
                                         warned)) is not None
    ]
    if not elements:
        return None

    return Storyboard(
        design_w=float(resolution.get('x', 1920.0) or 1920.0),
        design_h=float(resolution.get('y', 1080.0) or 1080.0),
        fit='min',
        elements=tuple(elements),
    )
