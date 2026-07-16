"""Bridge: harvested NotITG shader-flag events -> fullscreen shader passes.

# What a shader flag is

`GAMESTATE:SetShaderFlag(key)` / `SetShaderFlagNum(key, which)` toggle
NotITG's global shader-flag registry. Charts read those integer keys
back inside their own actor/screen shaders (`GetShaderFlag`), so a key's
*meaning* is chart-defined, not fixed by the engine - there is no
published key -> effect table (the craftedcart docs list the RageShader
API but not a flag catalogue). The classic template's `mod_shader(beat,
key, which)` helper sets `key` at `beat` and clears it (sets 0) 0.5
beats later, so each flag is a brief pulse.

# What we can honor

We ship two generic fullscreen passes whose look matches the flags that
real classic-template charts reach for (screen mirror / tiled-mirror
fold - the gat reference's kaleidoscoped playfield). `_FLAG_SHADERS`
maps the small set of flag keys observed in the local library's
`mod_shader` calls to those passes on a documented, best-effort basis.
Keys we cannot pin to a feasible fullscreen effect are SKIPPED (their
`which`/key is preserved in the returned skip list for logging), never
guessed - a wrong screen shader is worse than none.

# Output

`notitg_shader_effects` returns a one-element list holding a
`ShaderStackEffect` built from `.ffx`-shaped shader events, or []. Each
mapped flag pulse becomes a shader event that eases strength up at the
set beat and back to 0 at the clear beat, exactly the stack the fluXis
shader path already consumes.
"""
from __future__ import annotations

from analysis.player.render.shaders import ShaderStackEffect

# Flag key -> (shader id in the builtin library, params for u_strength.y/z).
# strength.x is driven by the on/off pulse; y/z carry the pass's mode
# knob (mirror axis, tile count). Keys are the ones the local NotITG
# library's `mod_shader` calls use; documented as best-effort until a
# real per-key oracle exists.
_FLAG_SHADERS = {
    53: ('screen_mirror', (0.0, 0.0)),   # horizontal fold
    54: ('screen_mirror', (1.0, 0.0)),   # vertical fold
    55: ('screen_tile', (2.0, 0.0)),     # 2x2 kaleidoscope
    48: ('screen_tile', (4.0, 0.0)),     # 4x4 tiling
}

# Keys seen locally that we deliberately do not map (no feasible
# fullscreen realization without the chart's own .frag / render targets).
_SKIPPED_KEYS = frozenset({49, 124, 217})

_PULSE_ON_MS = 40.0


def notitg_shader_effects(shader_flags) -> list:
    events, _skipped = build_shader_events(shader_flags)
    effect = ShaderStackEffect(events)
    return [effect] if effect else []


def build_shader_events(shader_flags):
    """(events, skipped_keys): `.ffx`-shaped shader events for the mapped
    flag pulses, plus the sorted list of distinct keys that were skipped.

    A flag key `k` set at time `t` turns its shader on; the next event
    that clears it (key 0, or the paired clear `mod_shader` emits) turns
    it off. Unpaired sets stay on until the chart's end (rare)."""
    flags = _clean(shader_flags)
    on_windows, skipped = _pair_pulses(flags)

    events = []
    for key, t_on, t_off in on_windows:
        shader_id, (mode_y, mode_z) = _FLAG_SHADERS[key]
        events.append({'shader': shader_id, 'time': t_on * 1000.0,
                       'duration': _PULSE_ON_MS, 'use-start': True,
                       'start-params': {'strength': 0.0},
                       'end-params': {'strength': 1.0, 'strength2': mode_y,
                                      'strength3': mode_z}})
        events.append({'shader': shader_id, 'time': t_off * 1000.0,
                       'duration': _PULSE_ON_MS,
                       'end-params': {'strength': 0.0, 'strength2': mode_y,
                                      'strength3': mode_z}})
    return events, sorted(skipped)


def _clean(shader_flags) -> list:
    out = []
    for row in shader_flags or []:
        if not isinstance(row, dict):
            continue
        key = row.get('key')
        t = row.get('t')
        if key is None or t is None:
            continue
        out.append((int(key), float(t)))
    out.sort(key=lambda r: r[1])
    return out


def _pair_pulses(flags):
    """Turn the (key, t) set/clear stream into on-windows. Key 0 clears
    whatever is currently on; a nonzero key sets that key on. Returns
    (windows, skipped_keys)."""
    windows = []
    skipped = set()
    open_since: dict = {}
    for key, t in flags:
        if key == 0:
            _close_all(open_since, t, windows)
            continue
        if key in _SKIPPED_KEYS or key not in _FLAG_SHADERS:
            skipped.add(key)
            continue
        if key not in open_since:
            open_since[key] = t
    _close_all(open_since, None, windows)
    return windows, skipped


def _close_all(open_since, t, windows) -> None:
    for key, t_on in open_since.items():
        t_off = t if t is not None else t_on + 0.5
        windows.append((key, t_on, t_off))
    open_since.clear()
