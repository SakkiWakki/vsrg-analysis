"""osu! storyboard (.osb + .osu [Events]) -> storyboard IR.

Storyboard scripting reference: objects are `Sprite,<layer>,<origin>,
"<file>",<x>,<y>` / `Animation,...,<frames>,<delay>[,<loop>]` (legacy
numeric aliases 4/6), followed by indented commands `<type>,<easing>,
<start>,<end>,<params...>`. Compilation notes:

- An empty end time means end = start; missing end params mean the
  value holds (end = start values). Zero-duration commands snap.
- Before its first command of a property, a sprite shows that
  command's START value (osu's documented behavior), so each
  property's rest state is the earliest command's start.
- `L` loops repeat their inner commands `count` times; inner times are
  relative to the loop start and one iteration spans the largest inner
  end time. Expanded at parse (capped defensively). `T` triggers are
  hitsound-driven and skipped, including their children.
- `P` parameters (H/V flip, A additive) apply for the whole element
  lifetime here; osu technically scopes them to the command window,
  but zero-duration usage (the overwhelmingly common form) already
  means "entire lifetime" and per-window flips are vanishingly rare.
- `R` rotation is radians (IR uses degrees); `C` color is 0-255.
- Coordinates live in osu's 640x480 space; 'height' fit reproduces
  the widescreen convention where wide viewports extend the x range.
- The Fail layer never shows in our pass-state playback and is
  dropped. .osb files may define `[Variables]` ($name=value),
  substituted textually before parsing.
"""
from __future__ import annotations

from pathlib import Path

from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.storyboard import Element, Storyboard
from analysis.player.render.storyboard.model import build_timelines

_DESIGN_W, _DESIGN_H = 640.0, 480.0

_LAYER_Z = {'background': -900, 'pass': -860, 'foreground': -820,
            'overlay': 700}
_LAYER_ALIASES = {'0': 'background', '1': 'fail', '2': 'pass',
                  '3': 'foreground', '4': 'overlay'}

_ORIGINS = {
    'topleft': (0.0, 0.0), 'centre': (0.5, 0.5), 'centreleft': (0.0, 0.5),
    'topright': (1.0, 0.0), 'bottomcentre': (0.5, 1.0),
    'topcentre': (0.5, 0.0), 'custom': (0.0, 0.0),
    'centreright': (1.0, 0.5), 'bottomleft': (0.0, 1.0),
    'bottomright': (1.0, 1.0),
}
_ORIGIN_ALIASES = {'0': 'topleft', '1': 'centre', '2': 'centreleft',
                   '3': 'topright', '4': 'bottomcentre', '5': 'topcentre',
                   '6': 'custom', '7': 'centreright', '8': 'bottomleft',
                   '9': 'bottomright'}

_OBJECT_HEADS = {'sprite': 'sprite', '4': 'sprite',
                 'animation': 'frames', '6': 'frames'}

_MAX_EXPANDED_COMMANDS = 100_000


class _RawObject:
    def __init__(self, kind, layer, origin, path, x, y, frames_spec):
        self.kind = kind
        self.layer = layer
        self.origin = origin
        self.path = path
        self.x = x
        self.y = y
        self.frames_spec = frames_spec   # (count, delay_ms, loop_forever)
        self.commands = []               # (type, easing, st_ms, et_ms, vals)
        self.flags = set()               # P params seen ('H','V','A')


def _split_quoted(line: str) -> list:
    """Comma split that keeps quoted segments (file paths) intact."""
    parts = []
    current = []
    quoted = False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == ',' and not quoted:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def _substitute_variables(lines: list) -> list:
    variables = {}
    in_vars = False
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('['):
            in_vars = stripped.lower() == '[variables]'
            continue
        if in_vars:
            if '=' in stripped and stripped.startswith('$'):
                name, _, value = stripped.partition('=')
                variables[name.strip()] = value.strip()
            continue
        out.append(line)
    if not variables:
        return out
    # Longest-first so $var10 substitutes before $var1.
    ordered = sorted(variables.items(), key=lambda kv: -len(kv[0]))
    substituted = []
    for line in out:
        if '$' in line:
            for name, value in ordered:
                line = line.replace(name, value)
        substituted.append(line)
    return substituted


def _events_lines(osu_path: Path) -> list:
    """[Events] lines of the .osu plus the whole sibling .osb (variables
    resolved), in that order (the .osb layers over the .osu per osu)."""
    lines = []
    try:
        text = osu_path.read_text(encoding='utf-8-sig', errors='replace')
    except OSError:
        return []
    in_events = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('['):
            in_events = stripped.lower() == '[events]'
            continue
        if in_events:
            lines.append(line)

    for osb in sorted(osu_path.parent.glob('*.osb')):
        try:
            osb_text = osb.read_text(encoding='utf-8-sig', errors='replace')
        except OSError:
            continue
        lines.extend(_substitute_variables(osb_text.splitlines()))
    return lines


def _depth(line: str) -> int:
    depth = 0
    for ch in line:
        if ch in ' _':
            depth += 1
        else:
            break
    return depth


def _float(value, fallback=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_object_head(parts) -> _RawObject | None:
    head = parts[0].strip().lower()
    kind = _OBJECT_HEADS.get(head)
    if kind is None or len(parts) < 6:
        return None
    layer = _LAYER_ALIASES.get(parts[1].strip(),
                               parts[1].strip().lower())
    origin_name = _ORIGIN_ALIASES.get(parts[2].strip(),
                                      parts[2].strip().lower())
    origin = _ORIGINS.get(origin_name, (0.0, 0.0))
    path = parts[3].strip().strip('"').replace('\\', '/')

    frames_spec = None
    if kind == 'frames':
        if len(parts) < 8:
            return None
        loop = parts[8].strip().lower() if len(parts) > 8 else 'loopforever'
        frames_spec = (max(1, int(_float(parts[6], 1))),
                       _float(parts[7], 0.0),
                       loop != 'looponce')
    return _RawObject(kind, layer, origin, path,
                      _float(parts[4]), _float(parts[5]), frames_spec)


def _command_values(ctype: str, params: list) -> tuple | None:
    """(start_vals, end_vals) tuples for one command; None when the
    command carries no animatable values (P) or is malformed."""
    values = [_float(p) for p in params]

    match ctype:
        case 'F' | 'MX' | 'MY' | 'S' | 'R':
            if not values:
                return None
            start = (values[0],)
            end = (values[1],) if len(values) > 1 else start
            return (start, end)
        case 'M' | 'V':
            if len(values) < 2:
                return None
            start = (values[0], values[1])
            end = ((values[2], values[3]) if len(values) >= 4 else start)
            return (start, end)
        case 'C':
            if len(values) < 3:
                return None
            start = tuple(v / 255.0 for v in values[:3])
            end = (tuple(v / 255.0 for v in values[3:6])
                   if len(values) >= 6 else start)
            return (start, end)
    return None


def _parse_command_line(parts) -> tuple | None:
    """(type, easing, st_ms, et_ms, start_vals, end_vals) or None."""
    ctype = parts[0].strip().lstrip(' _').upper()
    if len(parts) < 4:
        return None
    easing = int(_float(parts[1], 0.0))
    st = _float(parts[2], 0.0)
    et_raw = parts[3].strip()
    et = st if et_raw == '' else _float(et_raw, st)
    vals = _command_values(ctype, parts[4:])
    if vals is None:
        return None
    return (ctype, easing, st, max(st, et), vals[0], vals[1])


def _expand_loop(loop_start, count, inner) -> list:
    if not inner:
        return []
    iteration = max(et for _t, _e, _st, et, _s, _v in inner)
    if iteration <= 0:
        return []
    total = min(count, _MAX_EXPANDED_COMMANDS // max(1, len(inner)))
    out = []
    for i in range(max(1, total)):
        offset = loop_start + i * iteration
        out.extend((t, e, st + offset, et + offset, s, v)
                   for t, e, st, et, s, v in inner)
    return out


def _parse_objects(lines) -> list:
    objects = []
    current = None
    block = None       # ('L', start, count, [inner]) | ('T',) while active
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
        parts = _split_quoted(stripped)
        depth = _depth(line)

        if depth == 0:
            if block is not None and current is not None:
                _close_block(current, block)
            block = None
            current = _parse_object_head(parts)
            if current is not None:
                objects.append(current)
            continue
        if current is None:
            continue

        head = parts[0].strip().upper()
        if depth == 1 and block is not None:
            _close_block(current, block)
            block = None
        if head == 'L' and depth == 1:
            block = ('L', _float(parts[1], 0.0),
                     max(1, int(_float(parts[2], 1.0))), [])
            continue
        if head == 'T' and depth == 1:
            block = ('T',)
            continue

        if head == 'P':
            for flag in (parts[4].strip().upper() if len(parts) > 4 else ''):
                current.flags.add(flag)
            continue
        command = _parse_command_line(parts)
        if command is None:
            continue
        if block is not None and block[0] == 'L' and depth >= 2:
            block[3].append(command)
        elif block is not None and block[0] == 'T' and depth >= 2:
            continue
        else:
            current.commands.append(command)
    if block is not None and current is not None:
        _close_block(current, block)
    return objects


def _close_block(obj, block) -> None:
    if block[0] == 'L':
        obj.commands.extend(_expand_loop(block[1], block[2], block[3]))


_COMMAND_PROPS = {'F': ('alpha',), 'MX': ('x',), 'MY': ('y',),
                  'R': ('rotation',), 'S': ('scale_x', 'scale_y'),
                  'M': ('x', 'y'), 'V': ('scale_x', 'scale_y'),
                  'C': ('color',)}
_RADIANS_TO_DEGREES = 57.29577951308232


def _prop_keyframes(obj) -> tuple:
    """(keyframes dict, rests dict, lifetime) compiled from commands."""
    per_prop: dict = {}
    t_min = None
    t_max = None
    for ctype, easing, st, et, start_vals, end_vals in obj.commands:
        props = _COMMAND_PROPS.get(ctype)
        if props is None:
            continue
        t_min = st if t_min is None else min(t_min, st)
        t_max = et if t_max is None else max(t_max, et)
        for i, prop in enumerate(props):
            if ctype == 'C':
                start, end = start_vals, end_vals
            elif len(props) == 2 and ctype != 'S':
                start, end = (start_vals[i],), (end_vals[i],)
            elif ctype == 'S':
                start, end = (start_vals[0],), (end_vals[0],)
            elif ctype == 'R':
                start = (start_vals[0] * _RADIANS_TO_DEGREES,)
                end = (end_vals[0] * _RADIANS_TO_DEGREES,)
            else:
                start, end = start_vals, end_vals
            per_prop.setdefault(prop, []).append(Keyframe(
                t=st / 1000.0,
                values=end,
                duration=max(0.0, et - st) / 1000.0,
                easing=easing,
                start=start,
            ))

    keyframes = {}
    rests = {'x': obj.x, 'y': obj.y}
    for prop, frames in per_prop.items():
        frames.sort(key=lambda kf: kf.t)
        keyframes[prop] = frames
        rests[prop] = frames[0].start if len(frames[0].start) > 1 \
            else frames[0].start[0]
    return keyframes, rests, (t_min, t_max)


def _frame_paths(base_dir: Path, path: str, count: int) -> tuple:
    stem, dot, ext = path.rpartition('.')
    if not dot:
        stem, ext = path, ''
    return tuple(str(base_dir / f'{stem}{i}{dot}{ext}')
                 for i in range(count))


def _compile_object(obj, base_dir: Path) -> Element | None:
    z = _LAYER_Z.get(obj.layer)
    if z is None:
        return None
    keyframes, rests, (t_min, t_max) = _prop_keyframes(obj)
    if t_min is None:
        return None

    frames = ()
    frame_delay = 0.0
    loop_forever = True
    asset = str(base_dir / obj.path)
    if obj.kind == 'frames':
        count, delay_ms, loop_forever = obj.frames_spec
        frames = _frame_paths(base_dir, obj.path, count)
        frame_delay = delay_ms / 1000.0
        asset = None

    return Element(
        kind=obj.kind,
        z=z,
        z_index=0,
        t_start=t_min / 1000.0,
        t_end=t_max / 1000.0,
        anchor=(0.0, 0.0),
        origin=obj.origin,
        timelines=build_timelines(rests, keyframes),
        asset=asset,
        additive='A' in obj.flags,
        flip_h='H' in obj.flags,
        flip_v='V' in obj.flags,
        frames=frames,
        frame_delay=frame_delay,
        loop_forever=loop_forever,
    )


def parse_osu_storyboard(osu_path) -> Storyboard | None:
    """Compile the .osu's [Events] + sibling .osb into the storyboard
    IR; None when there are no drawable objects."""
    osu_path = Path(osu_path)
    objects = _parse_objects(_events_lines(osu_path))
    elements = [
        el for obj in objects
        if (el := _compile_object(obj, osu_path.parent)) is not None
    ]
    if not elements:
        return None
    return Storyboard(design_w=_DESIGN_W, design_h=_DESIGN_H,
                      fit='height', elements=tuple(elements))
