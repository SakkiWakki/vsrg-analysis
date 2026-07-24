"""NotITG / StepMania actor Lua API surface, one declarative registry.

This module is the SINGLE SOURCE OF TRUTH for every actor/GAMESTATE Lua
function a NotITG modfile can call. It has two layers:

- the VERB TABLES the recorder and the Lua stub bridge consume (setter /
  add / size / tween / getter maps, screen-constant resolution), plus the
  permissive-singleton bootstrap and recorder metatable (`_PERMISSIVE_BOOTSTRAP`)
  the engine-loop host runs - one copy, imported by the recorder and the sim;
- a declarative `VERB_REGISTRY` covering the COMPLETE called surface (the
  committed called-methods list beside this module + the ENGINE_ORACLE
  fork-additions list): every name maps to a `Verb(category, native,
  note, source)`,
  where `category` says how the engine treats it and `native` is the
  storyboard property / channel / handler it drives. A name is in exactly
  one of three coverage states - IMPLEMENTED (a category the recorder or
  a downstream pass acts on), IGNORED (cosmetic / no visual, with a
  one-line why), or DEFERRED (a real capability not yet built, with a
  why + the executor that will own it). tests/test_lua_api.py asserts the
  whole called surface resolves, so a chart calling an unmapped name
  fails CI instead of silently no-oping.

`source` on an entry is the engine file:line (from ENGINE_ORACLE.md) that
fixes the semantics when they are non-obvious - the effect oscillators
(Actor.cpp), the perspective/vanish point (ActorFrame.cpp), the AFT
capture (ActorFrameTexture.cpp).

Examples:
- `x`/`zoomto`/`linear`/`hidden` are SCALAR_SETTER / SIZE_SETTER /
  TWEEN / VISIBILITY - the recorder's tween model consumes them.
- `vibrate`/`wag`/`effectmagnitude` are EFFECT_OSCILLATOR - recorded as
  oscillator state (magnitude/period/clock), synthesized by the compiled
  document as an analytic channel (ENGINE_ORACLE GAP #3).
- `SetTextureFiltering` is IGNORED (cosmetic filter, no geometry).
- `EnableDepthBuffer` is DEFERRED (GL executor owns depth-buffered AFT
  captures).
"""
from __future__ import annotations

import re

from dataclasses import dataclass

from analysis.player.render.expr import ast as _expr_ast
from analysis.player.render.expr import eval_tree as _expr_eval
from analysis.player.render.expr import parser as _expr_parser
from analysis.player.render.expr import surface as _expr_surface

# Tween verb -> easing id (osu.Framework Easing enum shared by the
# storyboard timelines). SM's accelerate = ease-in quad, decelerate =
# ease-out quad, smooth = in-out cubic; linear/tween are linear.
_TWEEN_EASING = {
    'linear': 0, 'tween': 0, 'accelerate': 3, 'decelerate': 4,
    'smooth': 8,
}
_FALLBACK_TWEEN_EASING = 0

# Integration pokes closer together than this belong to one driving
# window; a real gap between perframe sections is seconds long, while
# in-window ticks are 1/60s apart.
_DRIVEN_SPAN_GAP = 0.5

# Setter verb -> storyboard property (or a tuple of properties it feeds,
# for uniform zoom). 'z'/'rotationx'/'rotationy' have no 2D-storyboard
# analogue; they still record so an actor's full poke stream is legible,
# just onto their own synthetic property names. `basezoom` is SM's
# separate pre-multiplier (the AFT copies set basezoomy(-1) to flip the
# upside-down capture); it records onto its own property so `zoom` can
# animate on top without clobbering it (the field producer folds the
# two).
_SCALAR_SETTERS = {
    'x': 'x', 'y': 'y', 'z': 'z',
    'zoom': ('scale_x', 'scale_y'), 'zoomx': 'scale_x', 'zoomy': 'scale_y',
    'zoomz': 'scale_z',
    'basezoom': ('base_scale_x', 'base_scale_y'),
    'basezoomx': 'base_scale_x', 'basezoomy': 'base_scale_y',
    'rotationz': 'rotation', 'rotationx': 'rotation_x',
    'rotationy': 'rotation_y',
    'diffusealpha': 'alpha', 'skewx': 'skew_x', 'skewy': 'skew_y',
}
_ADD_SETTERS = {'addx': 'x', 'addy': 'y', 'addz': 'z'}

# SM crop family: `croptop(f)`/`cropbottom(f)`/`cropleft(f)`/`cropright(f)`
# each hide a fraction (0..1) of the actor's texture from one edge before
# it is drawn (SM Actor::SetCrop*; a scrolling-reveal / wipe primitive).
# Recorded onto their own crop_* properties so the renderer can inset the
# drawn region; rest 0 = no crop, matching an untouched actor. Folded into
# _SCALAR_SETTERS so the recorder's tween model handles them like any other
# scalar (they animate under `linear`).
_CROP_SETTERS = {
    'croptop': 'crop_top', 'cropbottom': 'crop_bottom',
    'cropleft': 'crop_left', 'cropright': 'crop_right',
}
_SCALAR_SETTERS.update(_CROP_SETTERS)


def _live_reset_props(prop) -> tuple:
    return prop if isinstance(prop, tuple) else (prop,)


# verb -> (properties it resets, is-relative). Feeds `live_poke`'s
# accumulator re-anchor during a sampling pass: absolute setters replace
# the running value, `add*` setters offset it. Derived from the setter
# tables so the two stay in step.
_LIVE_RESET = {verb: (_live_reset_props(prop), False)
               for verb, prop in _SCALAR_SETTERS.items()}
_LIVE_RESET.update({verb: ((prop,), True)
                    for verb, prop in _ADD_SETTERS.items()})

# Absolute-size setters. SM's `zoomto(w, h)`/`setsize(w, h)` set the
# on-screen size in design pixels DIRECTLY (unlike `zoom`, a multiplier
# of the logical size). The renderer overrides natural*scale with these
# when they are set - the mechanism behind gat's fullscreen FUCK bars,
# `zoomto(20, SCREEN_HEIGHT)` on a 4px-wide sheet frame. Recorded onto
# their own size_x/size_y properties; the width-/height-only forms set
# one axis. Rest is the UNSET sentinel (negative), so an untouched size
# leaves the natural*scale path alone.
_SIZE_PAIR_SETTERS = frozenset({'zoomto', 'setsize'})
_SIZE_AXIS_SETTERS = {
    'zoomtowidth': 'size_x', 'setwidth': 'size_x',
    'zoomtoheight': 'size_y', 'setheight': 'size_y',
}
_SIZE_UNSET = -1.0

_REST = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
    'base_scale_x': 1.0, 'base_scale_y': 1.0, 'base_scale_z': 1.0,
    'rotation': 0.0, 'rotation_x': 0.0, 'rotation_y': 0.0,
    'alpha': 1.0, 'skew_x': 0.0, 'skew_y': 0.0,
    'zbias': 0.0,
    'size_x': _SIZE_UNSET, 'size_y': _SIZE_UNSET,
    'crop_top': 0.0, 'crop_bottom': 0.0, 'crop_left': 0.0, 'crop_right': 0.0,
    'color': (1.0, 1.0, 1.0),
    'frame': 0.0,
    'hidden': 0.0,
}

# Actor getter verb -> the property whose CURRENT value it returns.
# Per-frame driver closures read these (`source:GetY()`,
# `source:GetZoom()`) to feed sibling actors, so a getter must hand back
# a real number for that arithmetic to land as a keyframe instead of
# faulting on a table. Reads use the last set value (rest when never
# poked), ignoring in-flight tweens - a load-time snapshot.
_SCALAR_GETTERS = {
    'GetX': 'x', 'GetY': 'y', 'GetZ': 'z',
    'GetZoom': 'scale_x', 'GetZoomX': 'scale_x', 'GetZoomY': 'scale_y',
    'GetRotationX': 'rotation_x', 'GetRotationY': 'rotation_y',
    'GetRotationZ': 'rotation',
}


# SM screen constants that appear as classic-command args (`x,
# SCREEN_CENTER_X`). The Lua stub env resolves these for %function
# bodies; classic strings are parsed to raw tokens, so the recorder
# resolves them here. 640x480 design space, the engine's default.
_SCREEN_CONSTANTS = {
    'SCREEN_WIDTH': 640.0, 'SCREEN_HEIGHT': 480.0,
    'SCREEN_CENTER_X': 320.0, 'SCREEN_CENTER_Y': 240.0,
    'SCREEN_LEFT': 0.0, 'SCREEN_RIGHT': 640.0,
    'SCREEN_TOP': 0.0, 'SCREEN_BOTTOM': 480.0,
    'sw': 640.0, 'sh': 480.0,
}


def _as_float(value, default=None):
    # Hot path: almost every poke arg / read is already a Python float coming
    # out of frame_eval, so skip the float() call + exception frame for it
    # (~3M calls per heavy-chart bake). `type() is` is byte-identical to float()
    # for a genuine float/int and leaves the str/screen-expr path untouched.
    t = type(value)
    if t is float:
        return value
    if t is int:
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        resolved = _resolve_screen_expr(value.strip())
        if resolved is not None:
            return resolved
    return default


_SCREEN_SURFACE = _expr_surface.ConstSurface(_SCREEN_CONSTANTS)


def _resolve_screen_expr(text: str):
    """A screen constant with literal arithmetic around it
    (`SCREEN_CENTER_X`, `-SCREEN_WIDTH/2`, `112*(SCREEN_HEIGHT/480)`), or
    None when `text` neither is nor references a screen constant. Parses the
    arg once and evaluates it over the screen-constant surface; a bare number
    or a non-screen expression returns None so the caller falls through to
    plain float parsing."""
    constant = _SCREEN_CONSTANTS.get(text)
    if constant is not None:
        return constant
    node = _expr_parser.parse_guard(text)
    if node is None or not _references_screen(node):
        return None
    value = _expr_eval.tree_eval(node, _SCREEN_SURFACE)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _references_screen(node) -> bool:
    """True when the expression names at least one screen constant - the
    gate that keeps a plain literal from resolving here (it must fall
    through to float parsing)."""
    match node:
        case _expr_ast.Sym(name=name):
            return name in _SCREEN_CONSTANTS
        case _expr_ast.Unary(operand=operand):
            return _references_screen(operand)
        case _expr_ast.Binary(left=left, right=right):
            return _references_screen(left) or _references_screen(right)
        case _expr_ast.Index(base=base, key=key):
            return _references_screen(base) or _references_screen(key)
    return False


def _as_int(value, default=None):
    # Hot path: a rec_id / index arriving already int (the common case) skips
    # the _as_float round-trip. `type() is int` excludes bool (a rare, correct
    # exclusion: float(True)==1.0 still resolves via the slow path identically).
    if type(value) is int:
        return value
    f = _as_float(value)
    return int(f) if f is not None else default



# -- verb registry -----------------------------------------------------------
#
# Coverage state of a verb (its `state`): IMPLEMENTED means a recorder or
# a downstream compile pass acts on it; IGNORED means it has no visual we
# model (each entry carries a one-line why); DEFERRED means a genuine
# capability not yet built (why + the executor that will own it). These
# are the only three states the coverage test accepts.
IMPLEMENTED = 'implemented'
IGNORED = 'ignored'
DEFERRED = 'deferred'

# Category = how the ENGINE treats a verb, which fixes how we record it.
# The recorder dispatches on these; new categories are added only for a
# genuinely new engine behavior.
SCALAR_SETTER = 'scalar-setter'      # -> storyboard property (x, zoom, ...)
ADD_SETTER = 'add-setter'            # -> property accumulate (addx, ...)
SIZE_SETTER = 'size-setter'          # -> absolute size_x/size_y (zoomto, ...)
CROP_SETTER = 'crop-setter'          # -> crop_* inset (croptop, ...)
TWEEN_VERB = 'tween-verb'            # -> open easing interval (linear, ...)
TWEEN_CONTROL = 'tween-control'      # -> close/abandon interval (sleep, ...)
VISIBILITY = 'visibility'            # -> hidden bit (hidden, visible)
DIFFUSE = 'diffuse'                  # -> color + alpha (diffuse, ...)
SPRITE_STATE = 'sprite-state'        # -> sheet frame pin (setstate, animate)
EFFECT_OSCILLATOR = 'effect-osc'     # -> analytic oscillator channel (vibrate)
VANISH = 'vanish'                    # -> fov vanish center (SetVanishPoint)
GETTER = 'getter'                    # -> property read (GetX, ...)
COMMAND_DISPATCH = 'command'         # -> message/command run (playcommand, ...)
CAPTURE = 'capture'                  # -> AFT/proxy render-target surface
BLEND = 'blend'                      # -> additive/blend mode
SHADER = 'shader'                    # -> chart shader flags / programs
ENGINE_QUERY = 'engine-query'        # -> stubbed engine state read
NOTEFIELD = 'notefield'              # -> notefield draw-split control


@dataclass(frozen=True)
class Verb:
    """One entry in the actor Lua API surface.

    `category` is how the engine treats the call; `native` is the
    storyboard property / channel / handler it drives (a property name, a
    tuple of them, or a short handler tag), or None when the category
    needs no target; `note` is a one-line why for IGNORED / DEFERRED and a
    semantics note otherwise; `source` is the engine file:line that fixes
    non-obvious semantics (ENGINE_ORACLE.md), or ''."""
    name: str
    category: str
    state: str
    native: object = None
    note: str = ''
    source: str = ''


def _entries(names, category, state, native=None, note='', source=''):
    """A `{name: Verb}` fragment sharing one category/state, one per name
    in `names`. `native` may be a dict (per-name target) or a single value
    applied to all."""
    per_name = isinstance(native, dict)
    return {name: Verb(name, category, state,
                       native[name] if per_name else native, note, source)
            for name in names}


def _table_entries(table, category, state, note='', source=''):
    """A `{name: Verb}` fragment sourced FROM a ported verb table
    ({verb: property}), so the registry stays in step with the table the
    recorder actually consumes - no name is retyped."""
    return {verb: Verb(verb, category, state, prop, note, source)
            for verb, prop in table.items()}


# The registry, assembled FROM the ported verb tables (so the recorder and
# the registry cannot drift) plus declarative entries for verbs no table
# covered - the effect oscillators, vanish point, command dispatch,
# capture/proxy surface, blend, shaders, engine queries, and the
# ignored/deferred markers. Every name in the gat called surface + the
# ENGINE_ORACLE fork additions appears exactly once.
_A = 'Actor.cpp'  # effect oscillators (see ENGINE_ORACLE 2e)
VERB_REGISTRY: dict = {}

# Scalar / add / size / crop setters, tween verbs, getters - straight off
# the ported tables. `_CROP_SETTERS` is folded into `_SCALAR_SETTERS`, so
# it is re-tagged CROP_SETTER after the scalar pass.
VERB_REGISTRY.update(_table_entries(_SCALAR_SETTERS, SCALAR_SETTER, IMPLEMENTED))
VERB_REGISTRY.update(_table_entries(_CROP_SETTERS, CROP_SETTER, IMPLEMENTED,
                                    note='texture-edge inset fraction',
                                    source='Actor::SetCropTop et al.'))
VERB_REGISTRY.update(_table_entries(_ADD_SETTERS, ADD_SETTER, IMPLEMENTED))
VERB_REGISTRY.update(_table_entries(_SIZE_AXIS_SETTERS, SIZE_SETTER, IMPLEMENTED))
VERB_REGISTRY.update(_entries(_SIZE_PAIR_SETTERS, SIZE_SETTER, IMPLEMENTED,
                              native=('size_x', 'size_y')))
VERB_REGISTRY.update(_entries(_TWEEN_EASING, TWEEN_VERB, IMPLEMENTED))
VERB_REGISTRY.update(_table_entries(_SCALAR_GETTERS, GETTER, IMPLEMENTED))

# Tween-interval control (no channel value).
VERB_REGISTRY.update(_entries(
    ('sleep', 'finishtweening', 'stoptweening'), TWEEN_CONTROL, IMPLEMENTED,
    note='closes/abandons the open tween interval'))
VERB_REGISTRY.update(_entries(
    ('stopeffect',), TWEEN_CONTROL, IMPLEMENTED,
    note='clears the actor effect oscillator (SM StopEffect)', source=_A))

# Visibility, diffuse/color, sprite state.
VERB_REGISTRY.update(_entries(('hidden', 'visible'), VISIBILITY, IMPLEMENTED,
                              native='hidden'))
VERB_REGISTRY.update(_entries(('diffuse',), DIFFUSE, IMPLEMENTED,
                              native=('color', 'alpha')))
VERB_REGISTRY.update(_entries(('setstate', 'animate'), SPRITE_STATE, IMPLEMENTED,
                              native='frame'))
VERB_REGISTRY.update(_entries(('settext',), SPRITE_STATE, IMPLEMENTED,
                              native='text', note='bitmaptext element content'))

# Effect oscillators (ENGINE_ORACLE GAP #3): continuous self-animation
# parameterized by (magnitude, period, offset, clock). Recorded as
# oscillator state; the compiled document synthesizes the analytic channel
# (deterministic - clock is song time/beat; vibrate needs a seeded RNG).
VERB_REGISTRY.update(_entries(
    ('vibrate', 'wag', 'bob', 'bounce', 'spin'), EFFECT_OSCILLATOR, IMPLEMENTED,
    native={'vibrate': 'position', 'wag': 'rotation', 'bob': 'position',
            'bounce': 'position', 'spin': 'rotation'},
    note='analytic oscillator sampled at playback t', source=_A))
# The fork's complete effect-kind list. floorwag is a wag variant;
# pulse/pulseramp are zoom oscillators; the rainbow/diffuse*/glow*
# families oscillate color. The sim actor records spans for all of
# them; synthesis beyond the position/rotation kinds is pending a
# color/zoom oscillator channel.
_FORK = 'NotITG fork (Actor::PushSelf registration)'
VERB_REGISTRY.update(_entries(
    ('floorwag', 'pulse', 'pulseramp', 'rainbow', 'diffuseshift',
     'diffuseblink', 'diffuseramp', 'glowshift', 'glowblink', 'glowramp'),
    EFFECT_OSCILLATOR, DEFERRED, None,
    note='fork effect kinds - recorded as spans; synthesis pending',
    source=_FORK))
VERB_REGISTRY.update(_entries(
    ('effectmagnitude', 'effectperiod', 'effectclock', 'effectm',
     'effectdelay', 'effecttiming', 'effectcolor1', 'effectcolor2'),
    EFFECT_OSCILLATOR, IMPLEMENTED, note='oscillator parameters', source=_A))
# Modchart primitives (ACTOR_LUA_API.md category 00). aux scratch state
# is recorded as a live per-actor channel the getters read back;
# luaeffect (registered above) and tween's custom-Lua-easing argument
# remain the deferred live tier.
VERB_REGISTRY.update(_entries(
    ('aux', 'addaux'), SCALAR_SETTER, IMPLEMENTED, native='aux',
    note='per-actor scratch state (modchart primitive)', source=_FORK))
VERB_REGISTRY['getaux'] = Verb(
    'getaux', GETTER, IMPLEMENTED, 'aux',
    'scratch-state readback (luaeffect carries data through this)',
    _FORK)
VERB_REGISTRY['GetTweenTimeLeft'] = Verb(
    'GetTweenTimeLeft', GETTER, IMPLEMENTED, None,
    'seconds remaining in the tween queue', _FORK)
VERB_REGISTRY['GetSecsIntoEffect'] = Verb(
    'GetSecsIntoEffect', GETTER, IMPLEMENTED, 'effect_secs',
    'seconds into the effect period (drives perframe copy math)', _A)
VERB_REGISTRY['GetEffectMagnitude'] = Verb(
    'GetEffectMagnitude', GETTER, IMPLEMENTED, 'effect_magnitude',
    'current oscillator magnitude readback', _A)
VERB_REGISTRY['luaeffect'] = Verb(
    'luaeffect', EFFECT_OSCILLATOR, DEFERRED, None,
    'arbitrary per-frame Lua effect (SetEffectLua) - live channel tier',
    'Actor.cpp:763')

# Vanish point (ENGINE_ORACLE GAP #1): fov vanish center for the frame's
# perspective. Recorded onto vanish_x/vanish_y; the scene projection
# (field_3d/transform3d) consumes it.
VERB_REGISTRY['SetVanishPoint'] = Verb(
    'SetVanishPoint', VANISH, IMPLEMENTED, ('vanish_x', 'vanish_y'),
    'fov vanish center; default 320,240', 'ActorFrame.cpp:172')
# Field of view: an ActorFrame's `fov(deg)` sets a perspective camera
# (RageDisplay LoadMenuPerspective) that projects the frame AND ALL ITS
# CHILDREN; fov 0 = orthographic. Recorded onto the `fov` channel; the
# scene projection reads it (cascading to child instances) instead of
# the hardcoded 45 default. 602 charts use it; the 3D-heavy ones
# override 45 (fov 60/80).
VERB_REGISTRY['fov'] = Verb(
    'fov', VANISH, IMPLEMENTED, ('fov',),
    'perspective camera field of view (deg); projects the frame subtree',
    'ActorFrame.cpp / RageDisplay LoadMenuPerspective')
VERB_REGISTRY['GetRandomVanishTransform'] = Verb(
    'GetRandomVanishTransform', VANISH, DEFERRED, None,
    'randomized fov/vanish transform - scene projection frontier',
    'ActorFrame.cpp')

# Command / message dispatch - actor-side. The Lua bridge routes these to
# the message system (the __COMMAND set); queuemessage is queued the
# same way play/queuecommand are.
VERB_REGISTRY.update(_entries(
    ('playcommand', 'queuecommand', 'queuemessage'), COMMAND_DISPATCH,
    IMPLEMENTED, note='run/queue a named command or message on this actor'))
VERB_REGISTRY['Broadcast'] = Verb(
    'Broadcast', COMMAND_DISPATCH, IMPLEMENTED, None,
    'MESSAGEMAN:Broadcast - run <Name>MessageCommand on every actor')

# Capture / proxy / AFT surface. SetTexture/SetTextureName/GetTexture are
# recorded (they tag AFT copies); Create/Enable* configure the render
# target; SetTarget/GetTarget bind a proxy.
VERB_REGISTRY.update(_entries(
    ('SetTextureName', 'SetTexture'), CAPTURE, IMPLEMENTED, native='aft',
    note='AFT render-target / copy texture handoff',
    source='ActorFrameTexture.cpp'))
VERB_REGISTRY['GetTexture'] = Verb(
    'GetTexture', GETTER, IMPLEMENTED, 'aft',
    'returns the AFT capture marker a copy sprite feeds SetTexture',
    'ActorFrameTexture.cpp')


class AftTexture:
    """What `GetTexture()` returns for an AFT: the 'aft:<name>' marker
    (SetTexture unwraps it via `.marker`) plus the RageTexture size API
    (image = the design-space capture size, texture = the next-pow2
    allocation the engine pads to). A plain object, not a str subclass:
    lupa flattens str subclasses to Lua strings, and rigs call size
    getters ON the returned value to compute UV scale.

    Examples: Mdrqnxtagon's crumple Polygon reads GetImageWidth() /
    GetTextureWidth() (-> 0.625) and the height pair (-> 0.9375) to
    scale its vertex-grid UVs before SetDrawMode."""

    __slots__ = ('marker',)

    def __init__(self, marker: str):
        self.marker = marker

    def __repr__(self):
        return self.marker

    def GetImageWidth(self, _self=None):
        return 640.0

    def GetImageHeight(self, _self=None):
        return 480.0

    def GetTextureWidth(self, _self=None):
        return 1024.0

    def GetTextureHeight(self, _self=None):
        return 512.0
VERB_REGISTRY.update(_entries(
    ('SetTarget', 'GetTarget'), CAPTURE, IMPLEMENTED, native='proxy',
    note='ActorProxy target bind - copy re-renders the target',
    source='ActorProxy.cpp:18'))
VERB_REGISTRY.update(_entries(
    ('Create', 'EnablePreserveTexture'), CAPTURE, IMPLEMENTED, native='aft',
    note='AFT allocate / accumulate (PreserveTexture = feedback echo)',
    source='ActorFrameTexture.cpp:42'))
VERB_REGISTRY.update(_entries(
    ('EnableAlphaBuffer',), CAPTURE, IMPLEMENTED, native='aft',
    note='AFT alpha channel - captured via transparent window pixmap',
    source='ActorFrameTexture.cpp:42'))
VERB_REGISTRY.update(_entries(
    ('EnableDepthBuffer', 'EnableFloat'), CAPTURE, DEFERRED, native='aft',
    note='depth/float AFT buffers - GL executor (3D z-tested captures)',
    source='ActorFrameTexture.cpp:42'))

# `getrotation()` returns (rx, ry, rz); the recorder answers it specially
# (not a single scalar), so it is a getter the Lua bridge routes but lives
# outside _SCALAR_GETTERS.
VERB_REGISTRY['getrotation'] = Verb(
    'getrotation', GETTER, IMPLEMENTED, ('rotation_x', 'rotation_y', 'rotation'),
    'returns (rx, ry, rz) - copy sprites read z to mirror a spin')

# Actor-tree navigation getters return real recorders/stubs, not a scalar,
# so the Lua bridge handles them off the generic getter path (they must
# NOT join __GETTER). TREE_NAV keeps them mapped for coverage while
# excluded from GETTER_NAMES.
TREE_NAV = 'tree-nav'
VERB_REGISTRY.update(_entries(
    ('GetChild', 'GetTopScreen'), TREE_NAV, IMPLEMENTED, None,
    note='actor-tree navigation - real recorder targets'))

# Blend / additive.
VERB_REGISTRY['blend'] = Verb(
    'blend', BLEND, IMPLEMENTED, 'additive',
    'blend mode; additive is the flag we model (Element.additive)')

# Chart shader flags + notefield shader programs.
VERB_REGISTRY.update(_entries(
    ('SetShaderFlag', 'SetShaderFlagNum', 'GetShaderFlag'), SHADER, IMPLEMENTED,
    note='chart-defined shader toggles; shader_bridge maps a few keys'))
VERB_REGISTRY['GetShader'] = Verb(
    'GetShader', SHADER, IMPLEMENTED, None,
    'the actor GLSL program custom uniforms upload onto (notitg_compat)')
VERB_REGISTRY.update(_entries(
    ('uniform1f', 'uniform1i', 'uniform2f', 'uniform3f', 'uniform4f',
     'uniform1fv', 'uniform2fv', 'uniform3fv', 'uniform4fv'),
    SHADER, IMPLEMENTED, note='custom scalar/vec uniform upload (glUniformNf)'))
VERB_REGISTRY.update(_entries(
    ('uniformMatrix2fv', 'uniformMatrix3fv', 'uniformMatrix4fv'),
    SHADER, DEFERRED, None,
    note='matrix uniform upload - GL executor (not yet wired)'))
VERB_REGISTRY.update(_entries(
    ('GetArrowShader', 'GetReceptorShader', 'GetHoldShader',
     'GetArrowPathShader'), SHADER, DEFERRED, None,
    note='per-notefield-element GLSL bind - not consumed (gat uses none)'))
VERB_REGISTRY.update(_entries(
    ('ClearArrowShader', 'ClearReceptorShader', 'ClearHoldShader',
     'ClearShader'), SHADER, DEFERRED, None,
    note='per-notefield-element GLSL clear - not consumed'))

# Notefield draw-split controls (NotITG fork; gat does not use them).
VERB_REGISTRY.update(_entries(
    ('AddDrawSplit', 'DrawExtraPixelsLeft', 'DrawExtraPixelsRight',
     'DrawHoldHeadForTapsOnSameRow'), NOTEFIELD, DEFERRED, None,
    note='fine-grained notefield draw splitting - not consumed'))

# Engine-state queries: stubbed to benign constants at load.
VERB_REGISTRY.update(_entries(
    ('GetSongBeat', 'GetSongBeatNoOffset'), ENGINE_QUERY, IMPLEMENTED, None,
    note='current song beat - drives perframe math (integrator ticks)'))
VERB_REGISTRY.update(_entries(
    ('ApplyGameCommand', 'ApplyModifiers'), ENGINE_QUERY, IMPLEMENTED, None,
    note="per-frame mod injection ('mod,...') - the integrator input"))
VERB_REGISTRY.update(_entries(
    ('GetPreference', 'SetPreference', 'GetDisplayWidth', 'GetDisplayHeight',
     'IsPlayerEnabled', 'GetCurStageStats', 'GetCurrentSong', 'GetVendor',
     'KeyPress', 'SystemMessage', 'GetText', 'SetWidth', 'SetHeight'),
    ENGINE_QUERY, IMPLEMENTED, None,
    note='stubbed engine query/state - benign constant at load'))

# Explicitly ignored: cosmetic or no visual we model, each with the why.
VERB_REGISTRY['SetTextureFiltering'] = Verb(
    'SetTextureFiltering', ENGINE_QUERY, IGNORED, None,
    'texture min/mag filter - cosmetic, no geometry effect')
VERB_REGISTRY['customtexturerect'] = Verb(
    'customtexturerect', ENGINE_QUERY, IGNORED, None,
    'custom UV rect - our sheet cropping already sets UVs per frame')

# The perspective/mod family names that surface as tokens (space/distant
# are PlayerOptions perspective mods, not actor verbs; recorded through the
# mod channels, not the actor recorder).
VERB_REGISTRY.update(_entries(
    ('space', 'distant', 'randomvanish'), VANISH, DEFERRED, None,
    note='perspective mods - scene-projection frontier (mod channels)'))


def names_by_state(state: str) -> tuple:
    """Registered verb names in coverage `state`, sorted."""
    return tuple(sorted(n for n, v in VERB_REGISTRY.items() if v.state == state))


def names_by_category(category: str) -> tuple:
    return tuple(sorted(n for n, v in VERB_REGISTRY.items()
                        if v.category == category))


# Name lists the Lua stub bridge (_PERMISSIVE_BOOTSTRAP) builds its `__GETTER` /
# `__COMMAND` sets FROM, so those sets are generated here instead of
# hand-kept in the Lua bootstrap. `__GETTER` routes a call to the
# recorder's value-returning path (`__actor_get`); it is exactly the
# getters the recorder answers - the scalar getters, the AFT `GetTexture`
# marker, and the (rx,ry,rz) `getrotation`. Effect/tree-nav getters are
# mapped for coverage but NOT here: the recorder does not answer them off
# the generic path (they degrade to the permissive sentinel / a real
# recorder target), so adding them would hand a poke stream a wrong value.
GETTER_NAMES = tuple(sorted((*_SCALAR_GETTERS, 'GetTexture', 'getrotation')))

# The sim path answers two more getters the poke-recorder cannot:
# GetSecsIntoEffect (a live clock read under the actor's effect clock)
# and GetText (the settext round-trip charts abuse as a number store).
# Kept OFF GETTER_NAMES so the harvest path's routing is untouched; the
# sim substitutes its own set. Folds into GETTER_NAMES at cutover.
SIM_GETTER_NAMES = tuple(sorted(
    (*GETTER_NAMES, 'GetSecsIntoEffect', 'GetText', 'getaux',
     'GetTweenTimeLeft', 'GetNumChildren', 'GetNumTapsInRange',
     'GetNumVertices', 'GetXMLDir')))
# `__COMMAND` = the actor commands whose dispatch runs the actor's
# `<Name>Command` on its own recorder (`__actor_command`): playcommand runs
# it now, queuecommand after the pending tween. `queuemessage` is a message
# broadcast queued for later, NOT a per-actor command run - it is mapped
# for coverage but stays off this set so the Lua bridge keeps its distinct
# (currently permissive) routing.
_ACTOR_COMMAND_VERBS = frozenset({'playcommand', 'queuecommand'})
COMMAND_NAMES = tuple(sorted(
    n for n, v in VERB_REGISTRY.items()
    if v.category == COMMAND_DISPATCH and n in _ACTOR_COMMAND_VERBS))


def resolve(name: str) -> Verb | None:
    """The registry entry for a called method name, or None when the name
    is unmapped (the coverage test's failure signal)."""
    return VERB_REGISTRY.get(name)


# -- the Lua stub bridge ------------------------------------------------------
#
# The permissive-singleton bootstrap and the recorder metatable that route
# `a:x(100)` / `a:GetX()` / `a:playcommand('N')` to Python (`__actor_poke` /
# `__actor_get` / `__actor_command`). Both the engine-loop host (sim/env.py)
# run this SAME bootstrap - it is the shared recorder bridge, generated here
# from the one name-set source of truth (GETTER_NAMES / COMMAND_NAMES) so the
# bridge and the recorder cannot drift on which calls return a value vs run a
# command.


def _lua_name_set(names) -> str:
    """A Lua set literal (`{GetX=true, GetY=true}`) from registry name
    lists, so the bridge's `__GETTER` / `__COMMAND` sets are generated
    from the one source of truth instead of hand-kept in the bootstrap."""
    return '{' + ', '.join(f'{name}=true' for name in names) + '}'


# A metatable that makes any missing key return a callable/indexable dummy.
# Colon-calls on our python tables pass the table as arg 1, so the dummy
# ignores every argument. Chained access (A:B():C()) and field reads (A.x)
# both land back on a permissive value.
_PERMISSIVE_BOOTSTRAP = """
local function permissive()
    local t = {}
    local mt = {}
    mt.__index = function(_, _key) return permissive() end
    mt.__call = function(_, ...) return permissive() end
    setmetatable(t, mt)
    return t
end
_G.__permissive = permissive

function __make_singleton(overrides)
    local t = overrides or {}
    setmetatable(t, {__index = function(_, _key)
        return function(...) return permissive() end
    end})
    return t
end

-- A recording actor: every method call (`a:x(100)`, `a:linear(1)`) is
-- routed to Python via __actor_poke and returns the table so SM's
-- chained `a:linear(1):x(0)` keeps working. `id` ties it to a Python
-- recorder; the table is what an InitCommand self-assigns to a global,
-- so later closures poking that global hit the same recorder.
--
-- Getters (`a:GetX()`, `a:getrotation()`, `AFT:GetTexture()`) route to
-- __actor_get, which returns the recorder's current value(s) so driver
-- closures can read one actor to drive another (`b:zoomx(a:GetX())`)
-- without faulting on a table. __actor_get hands back the permissive
-- sentinel for getters we do not model (`GetChild()` etc.), so those
-- chains keep working as before.
local __GETTER = __GETTER_SET
-- Command-dispatch verbs an actor exposes to the message system:
-- `a:playcommand('Name')` runs the actor's <Name>Command now;
-- `a:queuecommand('Name')` runs it after the actor's pending tween
-- time. Both route to Python (__actor_command) with the recorder id so
-- the dispatched body records onto the same recorder as `self`.
local __COMMAND = __COMMAND_SET
function __make_recorder(id)
    local t = {__recorder_id = id}
    setmetatable(t, {__index = function(_, key)
        if __GETTER[key] then
            return function(_self, ...) return __actor_get(id, key) end
        end
        if __COMMAND[key] then
            return function(_self, name, ...)
                __actor_command(id, key, name)
                return t
            end
        end
        return function(_self, ...)
            __actor_poke(id, key, ...)
            return t
        end
    end})
    return t
end

-- The top screen is a recorder (the chart pokes it directly - the
-- `screen:effectmagnitude(..)` camera vibe and `GetTopScreen():zoom(..)`
-- per-frame zoom), so it records like any actor, but it ALSO answers
-- GetChild/GetTopScreen. Those return real recorder tables (a player, or
-- the screen itself) instead of a poke, so player fetches and chained
-- `GetTopScreen():GetChild(..)` keep working.
function __make_screen_recorder(id)
    local t = __make_recorder(id)
    local mt = getmetatable(t)
    local poke_index = mt.__index
    mt.__index = function(tbl, key)
        if key == 'GetChild' then
            return function(_self, name) return __screen_get_child(name) end
        end
        if key == 'GetTopScreen' then
            return function(_self) return t end
        end
        return poke_index(tbl, key)
    end
    return t
end

-- NotITG embeds Lua 5.0; these live under the LuaJIT (5.1) runtime as
-- their renamed forms. The template calls the 5.0 names.
if math.mod == nil then math.mod = math.fmod end
if table.getn == nil then table.getn = function(t) return #t end end
"""

# The `__GETTER` / `__COMMAND` routing sets are the registry's name lists,
# rendered as Lua set literals so the bridge and the recorder cannot drift
# on which calls return a value vs run a command.
_PERMISSIVE_BOOTSTRAP = _PERMISSIVE_BOOTSTRAP.replace(
    '__GETTER_SET', _lua_name_set(GETTER_NAMES)).replace(
    '__COMMAND_SET', _lua_name_set(COMMAND_NAMES))
