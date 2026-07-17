"""Bridge: harvested NotITG shaders -> fullscreen shader passes.

Two independent NotITG shader surfaces feed the fullscreen pipeline:

- SHADER FLAGS (`build_shader_events` / `notitg_shader_effects`): the
  classic-template `mod_shader` pulses of NotITG's built-in shader-flag
  registry, mapped to builtin library passes. gat uses this.
- MAP-SUPPLIED FRAGS (`chart_shader_effect`): a chart's own `.frag`
  files attached to a fullscreen ActorFrameTexture sprite (the
  `CatAFT.aft` / `CatAFT.sprite` post-process pattern), translated onto
  our contract by notitg_compat and driven by their per-frame uniform
  pokes. The Government Knows tier. See the map-supplied section below.

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

import re
from pathlib import Path

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import (EventTimeline,
                                                     keyframes_from_events)
from analysis.player.render.shaders import ShaderStackEffect, library
from analysis.player.render.shaders.library import notitg_compat

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


# ─────────────────────────────────────────────────────────────────────
# Map-supplied fragment shaders (tier 2)
# ─────────────────────────────────────────────────────────────────────
#
# A Government-Knows-style chart attaches its own `.frag` to a fullscreen
# sprite that draws a whole-screen ActorFrameTexture:
#
#     <Layer Type="Sprite" OnCommand="CatAFT.aft(self,'x')"/>       -- render scene to AFT 'x'
#     <Layer Type="Sprite" Frag="shaders/vhs.frag"                   -- post-process it
#            OnCommand="CatAFT.sprite(self,'x'); self:GetShader():uniform1f('time', t) ..."/>
#
# For that fullscreen case the AFT is exactly our capture, so the frag
# maps onto our contract (notitg_compat.translate): sampler0 -> u_tex,
# imageCoord/textureCoord -> the fullscreen UV, and the chart's own
# scalar uniforms (`GetShader():uniform1f('phase', v)` pokes) become
# named custom uniforms the pipeline sets per frame. Per-actor frags
# (vertex-stage note deformers, engine-texture noise samplers) are NOT
# fullscreen-expressible; translate raises and we skip them (Stage B).
#
# COMPILED-DICT CONTRACT (`compiled['chart_shaders']`): a list of
#
#     {'name':      unique id stem (str),
#      'frag':      the raw NotITG GLSL source (str),   # or 'frag_path'
#      'frag_path': absolute path to the .frag (str),   # alternative to 'frag'
#      'uniforms':  {uniform_name: [<.ffx-shaped event dicts>], ...},
#      'windows':   [<.ffx-shaped on/off event dicts>]}  # optional; 'strength'
#                    > 0 means the pass is live (default: always live).
#
# `uniforms` streams and `windows` are what a recorder must harvest from
# the per-frame `GetShader():uniform1f(...)` / visibility pokes. Until
# that harvest exists (see the recorder note in the project memory), the
# bridge drives whatever static data is present and RESTS EVERY UNSET
# UNIFORM AT 0, so a frag authored as a no-op at 0 stays identity.

# The only sampler a fullscreen pass can be fed: the capture (u_tex). A
# translated frag that still declares any other sampler needs an engine
# texture we do not supply (noise/atlas tables, a second feedback
# capture) -- a per-actor (Stage-B) shader, skipped rather than run
# black. (u_tex2, the pre-chain capture, is not produced by the current
# translator, so it counts as unfeedable here too.)
_SAMPLER_DECL_RE = re.compile(r'\buniform\s+sampler2D\s+(\w+)\s*;')


def chart_shader_effect(chart_shaders):
    """Register each map-supplied frag and return a `ChartShaderEffect`
    driving them, or None when none are fullscreen-expressible. Skips
    (does not raise on) Stage-B per-actor frags."""
    passes = _build_chart_passes(chart_shaders)
    return ChartShaderEffect(passes) if passes else None


def _build_chart_passes(chart_shaders):
    """(`shader_id`, uniform-timelines, window-timeline) per registerable
    frag. A frag notitg_compat cannot translate to a fullscreen pass is
    skipped; only uniforms the translated shader actually declares are
    driven (a stray poke stream for an undeclared name is dropped)."""
    passes = []
    for entry in chart_shaders or []:
        built = _build_one_pass(entry)
        if built is not None:
            passes.append(built)
    return passes


def _build_one_pass(entry):
    """One `(shader_id, timelines, window)` pass, or None if the entry is
    malformed or its frag is a Stage-B (per-actor) shader."""
    if not isinstance(entry, dict):
        return None
    glsl, name = _frag_source(entry), entry.get('name')
    if not glsl or not name:
        return None
    shader_id = _register(name, glsl)
    if shader_id is None:
        return None
    declared = set(library.registered_uniform_names(shader_id))
    timelines = {uname: _stream_timeline(events)
                 for uname, events in (entry.get('uniforms') or {}).items()
                 if uname in declared}
    return shader_id, timelines, _window_timeline(entry.get('windows'))


def _stream_timeline(events) -> EventTimeline:
    """A single-value `EventTimeline` from `.ffx`-shaped events, resting
    at 0 (identity when the driving stream is absent)."""
    return EventTimeline(
        keyframes_from_events(events, ('strength',), (0.0,)), rest=(0.0,))


def _register(name, glsl):
    """Register `glsl` as a chart shader, returning its id or None for a
    Stage-B frag (no sampler0 / needs an engine texture we cannot feed)
    or a duplicate/invalid name."""
    try:
        contract = notitg_compat.translate(glsl)
    except ValueError:
        return None
    if _needs_unfeedable_texture(contract):
        return None
    return library.register_source(f'chart:notitg:{name}', contract)


def _needs_unfeedable_texture(contract_glsl) -> bool:
    samplers = set(_SAMPLER_DECL_RE.findall(contract_glsl))
    return bool(samplers - {'u_tex'})


def _frag_source(entry):
    """The frag GLSL for an entry: inline `frag`, else the file at
    `frag_path`, else None (missing / unreadable)."""
    if entry.get('frag'):
        return entry['frag']
    path = entry.get('frag_path')
    if not path:
        return None
    try:
        return Path(path).read_text(encoding='utf-8')
    except OSError:
        return None


def _window_timeline(windows):
    """A 0/1 liveness timeline from on/off events, or None when the pass
    is always live (no windows harvested)."""
    return _stream_timeline(windows) if windows else None


class ChartShaderEffect:
    """Per-frame fullscreen passes for a chart's own frags: each live
    pass emits `(shader_id, {uniform: value, ...})`, the custom uniforms
    sampled from their driving streams (0 at rest). A pass with a window
    timeline is emitted only while that window is live."""

    def __init__(self, passes):
        self._passes = tuple(passes)

    def __bool__(self):
        return bool(self._passes)

    def at(self, ctx) -> EffectFrame | None:
        t = ctx.t_now
        out = []
        for shader_id, timelines, window in self._passes:
            if window is not None and window.sample(t)[0] <= 0.0:
                continue
            uniforms = {name: tl.sample(t)[0]
                        for name, tl in timelines.items()}
            out.append((shader_id, uniforms))
        if not out:
            return None
        return EffectFrame(shaders=tuple(out))
