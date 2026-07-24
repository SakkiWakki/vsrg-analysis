"""Bridge: NotITG shader-flag pulses -> fullscreen shader passes.

The classic-template `mod_shader` pulses of NotITG's built-in
shader-flag registry, mapped to builtin library passes
(`build_shader_events` / `notitg_shader_effects`). gat uses this.

A chart's own `Frag=` .frag files are NOT a fullscreen concern: every
one is a per-actor program over the actor's own texture
(Sprite::DrawPrimitives, sampler0 = m_pTexture), so they compile as
shaded field-instance blits of their source AFT's at-position capture
slot (sim/producers frag payload -> gl_capture._frag_program). A
finished-frame post pass can never express them faithfully: the AFT
curtain idiom draws a black quad between the capture node and the
sampler, so a pass sampling the finished frame sees the curtain
(gat 2's MonitorOn window rendered pass(black)=black).

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

    A flag key `k` written to a slot at time `t` turns its shader on; the
    write that closes it (a `0` or a different key into the same slot)
    turns it off. A genuinely unclosed set persists to the chart's end -
    the chart's own oversight, never a fabricated off time."""
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
        if t_off is None:
            continue
        events.append({'shader': shader_id, 'time': t_off * 1000.0,
                       'duration': _PULSE_ON_MS,
                       'end-params': {'strength': 0.0, 'strength2': mode_y,
                                      'strength3': mode_z}})
    return events, sorted(skipped)


def _clean(shader_flags) -> list:
    """(slot, key, t) triples sorted by time. `which` is the slot index
    (`SetShaderFlagNum`'s second arg); plain `SetShaderFlag(key)` is
    slot 0. The slot is what closes a flag - preserving it is what lets a
    same-slot rewrite window-close the previous pass (the lifetime rule);
    dropping it merged every flag into one latch that only key 0 closed."""
    out = []
    for row in shader_flags or []:
        if not isinstance(row, dict):
            continue
        key = row.get('key')
        t = row.get('t')
        if key is None or t is None:
            continue
        out.append((int(row.get('which') or 0), int(key), float(t)))
    out.sort(key=lambda r: r[2])
    return out


def _pair_pulses(flags):
    """Turn the (slot, key, t) write stream into on-windows, honoring
    NotITG's single-value-per-slot registry: each `SetShaderFlag(Num)`
    write REPLACES that slot, so writing any new value (0 or another key)
    closes whatever the slot held. Independent slots stay simultaneously
    live. Returns (windows, skipped_keys); an unclosed window carries
    t_off None (persists - no fabricated close)."""
    windows = []
    skipped = set()
    open_in_slot: dict = {}
    for slot, key, t in flags:
        if open_in_slot.get(slot, (None,))[0] == key:
            continue
        _close_slot(open_in_slot, slot, t, windows)
        if key == 0:
            continue
        if key in _SKIPPED_KEYS or key not in _FLAG_SHADERS:
            skipped.add(key)
            continue
        open_in_slot[slot] = (key, t)
    for slot in list(open_in_slot):
        _close_slot(open_in_slot, slot, None, windows)
    return windows, skipped


def _close_slot(open_in_slot, slot, t, windows) -> None:
    """Close the flag currently held in `slot` (if any) at time `t`,
    emitting its (key, t_on, t_off) window. t None = no close observed
    (the window persists to the chart's end)."""
    held = open_in_slot.pop(slot, None)
    if held is not None:
        key, t_on = held
        windows.append((key, t_on, t))
