"""The full NotITG actor verb surface, GENERATED from mechanism tables.

The fork registers ~221 Lua methods on every actor (actor_api_names.txt),
but they collapse into a dozen mechanisms: the binary proves it - the
`*2` / bulk / per-axis families are identical binding stubs, differing
only in which slot they write. So this module does not hand-write 221
rows. It writes the AXIS lists and the MECHANISM patterns, then produces
the scalar / add / bulk / getter rows programmatically. Only rows a
pattern cannot express (the deferred `*2` and rotation-order traps, the
ignored render-state hints) are written by hand, each with a why.

Output shapes EXTEND the lua_api table formats so SimActor dispatch
consumes them with a minimal diff:

- `SCALAR_SETTERS`   verb -> prop (or a tuple of props for a uniform set)
- `ADD_SETTERS`      verb -> prop            (adds onto the destination)
- `BULK_SETTERS`     verb -> (prop, ...)     (positional multi-write)
- `GETTERS`          verb -> (prop, read)    (read = 'current' | 'dest')
- `IGNORED`/`DEFERRED` verb -> reason        (no visual / not yet built)

Semantics are pinned to sources, never guessed:
- stock verbs cite openitg refs/notitg/openitg-src/src/Actor.h (the
  LunaActor ADD_METHOD block, file:line);
- the reads split GetX/GetY/GetZ (m_current, Actor.h:107-109) from the
  Get*X dest reads (DestTweenState, Actor.h:110-158);
- fork-only families whose second-slot / rotation-order semantics the
  decompile cannot fix (Actor.clean.c is COMDAT-folded - string keys and
  addresses are trustworthy, adjacent callee names are not) are DEFERRED
  with the reason, not mapped to a guess.

The lua_api.VERB_REGISTRY stays the CI-firewalled coverage contract for
the gat CALLED surface; this module is the broader ENGINE surface (every
registered name), consumed by the sim and checked by
tests/test_verb_surface.py against actor_api_names.txt.
"""
from __future__ import annotations

# -- read kinds --------------------------------------------------------------
# A getter reads either the in-flight interpolated value (m_current) or the
# destination (the queue tail / DestTweenState). GetX/GetY/GetZ read
# current (Actor.h:107-109); every Get*X/Get*Y zoom/rotation getter reads
# the destination (Actor.h:144-158). The fork's GetCurrent* family reads
# current (ACTOR_LUA_API.md category 04/05).
READ_CURRENT = 'current'
READ_DEST = 'dest'

# openitg LunaActor binding block; the file:line an entry cites is the
# ADD_METHOD / accessor that fixes its semantics.
_H = 'openitg Actor.h'


# -- axis -> storyboard property ---------------------------------------------
# The property names SimActor / the storyboard element compiler already
# use (lua_api._REST is the authority on rest values). Rotation z is the
# 2D `rotation`; x/y are the synthetic 3D axes.
_POS = {'x': 'x', 'y': 'y', 'z': 'z'}
_ZOOM = {'x': 'scale_x', 'y': 'scale_y', 'z': 'scale_z'}
_BASEZOOM = {'x': 'base_scale_x', 'y': 'base_scale_y'}
_ROT = {'x': 'rotation_x', 'y': 'rotation_y', 'z': 'rotation'}
_SKEW = {'x': 'skew_x', 'y': 'skew_y'}


# -- mechanism 1: dest-state scalar writes -----------------------------------
# verb -> property (or tuple of properties for a uniform set). Straight
# writes onto the destination tween state (SetX -> DestTweenState().pos.x,
# Actor.h:113). `zoom` sets both scale axes (Actor.h:148).
SCALAR_SETTERS: dict = {
    'x': _POS['x'], 'y': _POS['y'], 'z': _POS['z'],
    'zoom': (_ZOOM['x'], _ZOOM['y']),
    'zoomx': _ZOOM['x'], 'zoomy': _ZOOM['y'], 'zoomz': _ZOOM['z'],
    'basezoomx': _BASEZOOM['x'], 'basezoomy': _BASEZOOM['y'],
    'rotationx': _ROT['x'], 'rotationy': _ROT['y'], 'rotationz': _ROT['z'],
    'diffusealpha': 'alpha',
    'skewx': _SKEW['x'],
    'zbias': 'zbias',
    'croptop': 'crop_top', 'cropbottom': 'crop_bottom',
    'cropleft': 'crop_left', 'cropright': 'crop_right',
}

# `basezoom`/`basezoomz`/`skewy` are fork additions whose stock analogue
# exists (SetBaseZoomX/SetSkewX pattern, Actor.h:495/166); the parser
# calls basezoom(f) as the uniform pre-multiplier both AFT flip axes use.
SCALAR_SETTERS['basezoom'] = (_BASEZOOM['x'], _BASEZOOM['y'])
SCALAR_SETTERS['basezoomz'] = 'base_scale_z'
SCALAR_SETTERS['skewy'] = _SKEW['y']


# -- mechanism 2: relative adds ----------------------------------------------
# AddX(v) = SetX(GetDestX()+v) - onto the DESTINATION (Actor.h:117). The
# per-axis rotation adds (addrotationx/y/z) are the fork's direct
# component adds (ACTOR_LUA_API.md category 04); addaux offsets scratch
# state. The composite addrotationxyz is a bulk add (mechanism 3).
ADD_SETTERS: dict = {
    'addx': _POS['x'], 'addy': _POS['y'], 'addz': _POS['z'],
    'addrotationx': _ROT['x'], 'addrotationy': _ROT['y'],
    'addrotationz': _ROT['z'],
}


# -- mechanism 3: bulk expansion -> N scalar writes --------------------------
# One call sets several axes positionally (ACTOR_LUA_API.md 03/04: "set
# several axes in one call"). verb -> the property tuple, in argument
# order. `xywh` sets position then size (the fork Polygon convenience);
# `xyza` appends alpha. rotationxyz sets the three rotation axes.
BULK_SETTERS: dict = {
    'xy': (_POS['x'], _POS['y']),
    'xyz': (_POS['x'], _POS['y'], _POS['z']),
    'xyza': (_POS['x'], _POS['y'], _POS['z'], 'alpha'),
    'xywh': (_POS['x'], _POS['y'], 'size_x', 'size_y'),
    'rotationxyz': (_ROT['x'], _ROT['y'], _ROT['z']),
}

# The bulk relative-add: three component adds in one call.
BULK_ADD_SETTERS: dict = {
    'addrotationxyz': (_ROT['x'], _ROT['y'], _ROT['z']),
}

# Crop composites: convenience forms that fan one call across the four
# scalar crop edges (already implemented as croptop/cropbottom/cropleft/
# cropright above, and the storyboard renderer already insets by them).
# `crop(l,t,r,b)` sets all four in openitg's left/top/right/bottom order;
# `croph(l,r)` is the horizontal pair, `cropv(t,b)` the vertical pair. Each
# entry lists its target crop props in positional argument order, so the
# expansion is the same positional zip the bulk setters use.
CROP_COMPOSITES: dict = {
    'crop': ('crop_left', 'crop_top', 'crop_right', 'crop_bottom'),
    'croph': ('crop_left', 'crop_right'),
    'cropv': ('crop_top', 'crop_bottom'),
}


# -- mechanism 1b: absolute-size setters -------------------------------------
# zoomto(w, h) / zoomtowidth(w) / zoomtoheight(h) set the on-screen size in
# design pixels directly (ZoomTo -> SetZoomX(w/naturalWidth), Actor.h:491;
# the sim records the target size onto size_x/size_y and the renderer
# overrides natural*scale with it). Pair form writes both; the width-/
# height-only forms write one axis.
SIZE_PAIR_SETTERS: dict = {
    'zoomto': ('size_x', 'size_y'),
}
SIZE_AXIS_SETTERS: dict = {
    'zoomtowidth': 'size_x',
    'zoomtoheight': 'size_y',
}


# -- mechanism 6: getters (dest vs current read) -----------------------------
# verb -> (property, read-kind). GetX/GetY/GetZ read m_current
# (Actor.h:107-109); the zoom/rotation/basezoom getters read the
# destination (Actor.h:144-158). The fork GetCurrent* family reads current
# (ACTOR_LUA_API.md 04/05). getrotation/getcurrentrotation return the
# rotation triple and are handled specially by SimActor.read (below).
GETTERS: dict = {
    'GetX': (_POS['x'], READ_CURRENT),
    'GetY': (_POS['y'], READ_CURRENT),
    'GetZ': (_POS['z'], READ_CURRENT),
    'GetZoom': (_ZOOM['x'], READ_DEST),
    'GetZoomX': (_ZOOM['x'], READ_DEST),
    'GetZoomY': (_ZOOM['y'], READ_DEST),
    'GetZoomZ': (_ZOOM['z'], READ_DEST),
    'GetBaseZoomX': (_BASEZOOM['x'], READ_DEST),
    'GetBaseZoomY': (_BASEZOOM['y'], READ_DEST),
    'GetBaseZoomZ': ('base_scale_z', READ_DEST),
    'GetRotationX': (_ROT['x'], READ_DEST),
    'GetRotationY': (_ROT['y'], READ_DEST),
    'GetRotationZ': (_ROT['z'], READ_DEST),
    'GetCurrentZoomX': (_ZOOM['x'], READ_CURRENT),
    'GetCurrentZoomY': (_ZOOM['y'], READ_CURRENT),
    'GetCurrentZoomZ': (_ZOOM['z'], READ_CURRENT),
    'GetCurrentRotationX': (_ROT['x'], READ_CURRENT),
    'GetCurrentRotationY': (_ROT['y'], READ_CURRENT),
    'GetCurrentRotationZ': (_ROT['z'], READ_CURRENT),
    'GetSkewX': (_SKEW['x'], READ_DEST),
    'GetSkewY': (_SKEW['y'], READ_DEST),
    'getaux': ('aux', READ_CURRENT),
}

# Getters returning a tuple, not a scalar - SimActor.read answers these on
# its own path. getrotation reads dest (Actor.h:523: GetRotationX/Y/Z),
# getcurrentrotation reads current.
TUPLE_GETTERS: dict = {
    'getrotation': ((_ROT['x'], _ROT['y'], _ROT['z']), READ_DEST),
    'getcurrentrotation': ((_ROT['x'], _ROT['y'], _ROT['z']), READ_CURRENT),
    'geteffectmagnitude': (None, READ_CURRENT),
}


# -- mechanism 12: explicit ignore / defer -----------------------------------
# IGNORED: registered, but no visual the sim models - each with the why.
IGNORED: dict = {
    'SetTextureFiltering': 'texture min/mag filter - cosmetic, no geometry',
    'SetTextureWrapping': 'texture wrap mode - cosmetic, no geometry',
    'texturefiltering': 'texture min/mag filter - cosmetic, no geometry',
    'texturewrapping': 'texture wrap mode - cosmetic, no geometry',
    'hibernate': 'defers first update; the sim already anchors load time',
    'shadowlength': 'drop-shadow offset - decorative, not composited',
    'draworder': 'sibling draw order within a frame - the element compiler '
                 'orders by z; no per-actor override consumed',
    'clearbuffer': 'per-frame color-buffer clear hint - render-target detail',
    'clearzbuffer': 'per-frame depth-buffer clear hint - render-target detail',
    'zbuffer': 'depth-buffer enable - z compositing is not modeled 2D',
    'ztest': 'depth test toggle - see zbuffer',
    'ztestmode': 'depth test mode string - see zbuffer',
    'zwrite': 'depth write toggle - see zbuffer',
    'backfacecull': 'backface cull toggle - single-sided 2D sprites',
    'cullmode': 'cull mode string - see backfacecull',
    'Reset': 'resets actor to its XML defaults - load-time only, no anim',
    'Draw': 'imperative re-draw into the current frame; the copy producer '
            'models re-draw through SetTarget binds, not this call',
    'GetXMLDir': 'actor XML source dir - asset resolution, not transform',
    'GetName': 'actor name string - identity, resolved by the env registry',
    'SetName': 'renames the actor - identity, no transform',
    'GetHidden': 'visibility readback - the sim answers GetHidden via '
                 'its own read path when a chart needs it',
    'customtexturerect': 'custom UV rect - sheet cropping already sets UVs',
    'align': 'shorthand for halign+valign - anchor, set at XML load',
    'halign': 'horizontal anchor fraction - load-time layout, not animated',
    'valign': 'vertical anchor fraction - load-time layout, not animated',
    'horizalign': 'horizontal align string - load-time layout',
    'vertalign': 'vertical align string - load-time layout',
}

# DEFERRED: a real capability, not yet built - each with the why + the
# executor that will own it. The `*2` second-slot family and the
# rotation-order verbs are the known traps: Actor.clean.c is COMDAT-folded
# (trust string keys, not callee names), so their semantics cannot be
# pinned from the decompile, and openitg has no analogue to copy. They are
# deferred, never guessed.
# The `*2` slot IS a second independent transform channel composed on top
# of the base one, resting at identity (pos/rot/skew 0, zoom 1) - proven
# by chart usage (Puuro "Poison Cupcake" sets xy AND xy2(0,0) on one
# actor, oscillates rotationx2/x2 about a held base pos). What is NOT
# recoverable is the COMPOSE ORDER: Actor::BeginDraw (@004a531c) builds
# the matrix but its body is COMDAT-folded to Screen*::TweenOffScreen
# (Actor.clean.c cheat-sheet, lines 33-73), so the base translate/rotate/
# zoom/skew push order and where the `2` block plugs in are gone; the
# object-layout map does not enumerate the `2` sub-fields. openitg has no
# analogue (Actor.h TweenState has ONE pos/rotation/scale, no `2`), and
# no `*2`-using chart has a reference frame (gat 1/gat 2 use zero `*2`
# verbs), so a guessed compose could not be parity-validated. Unblock via
# raw disasm of BeginDraw @004a531c or a captured Poison-Cupcake frame.
# Full dig: scratchpad/second_slot_findings.md.
_SECOND_SLOT = ("fork second-value slot ('<prop>2'): a second transform "
                "channel composed on the base (rest = identity), but the "
                "compose ORDER is unrecoverable - Actor::BeginDraw @004a531c "
                "is COMDAT-folded to Screen*::TweenOffScreen and openitg has "
                "no analogue; no *2-using chart has a reference frame to "
                "validate against (see second_slot_findings.md)")
_ROT_ORDER = ('order-dependent 3D rotation compose; needs a rotation-order '
              'model the 2D storyboard does not have (scene-projection tier)')
DEFERRED: dict = {
    # '*2' second-value slots (position/zoom/rotation/skew).
    'x2': _SECOND_SLOT, 'y2': _SECOND_SLOT, 'z2': _SECOND_SLOT,
    'xy2': _SECOND_SLOT, 'xyz2': _SECOND_SLOT,
    'zoom2': _SECOND_SLOT, 'zoomx2': _SECOND_SLOT, 'zoomy2': _SECOND_SLOT,
    'zoomz2': _SECOND_SLOT, 'zoomxyz2': _SECOND_SLOT,
    'rotationx2': _SECOND_SLOT, 'rotationy2': _SECOND_SLOT,
    'rotationz2': _SECOND_SLOT, 'rotationxyz2': _SECOND_SLOT,
    'skewx2': _SECOND_SLOT, 'skewy2': _SECOND_SLOT,
    'GetX2': _SECOND_SLOT, 'GetY2': _SECOND_SLOT, 'GetZ2': _SECOND_SLOT,
    'GetZoomX2': _SECOND_SLOT, 'GetZoomY2': _SECOND_SLOT,
    'GetZoomZ2': _SECOND_SLOT,
    'GetRotationX2': _SECOND_SLOT, 'GetRotationY2': _SECOND_SLOT,
    'GetRotationZ2': _SECOND_SLOT, 'GetRotation2': _SECOND_SLOT,
    'GetSkewX2': _SECOND_SLOT, 'GetSkewY2': _SECOND_SLOT,
    'getrotation2': _SECOND_SLOT,
    # rotation order + spherical heading/pitch/roll.
    'SetRotationOrder': _ROT_ORDER, 'GetRotationOrder': _ROT_ORDER,
    'heading': 'AddRotationH - spherical heading add, ' + _ROT_ORDER,
    'pitch': 'AddRotationP - spherical pitch add, ' + _ROT_ORDER,
    'roll': 'AddRotationR - spherical roll add, ' + _ROT_ORDER,
    # skew-before-vs-after-rotation is a transform-order control.
    'skewx_before_rotation': 'fork skew-order flag; needs the transform-order '
                             'model the storyboard lacks',
    'skewy_before_rotation': 'fork skew-order flag; needs the transform-order '
                             'model the storyboard lacks',
    'skewto': 'fork skew convenience (both axes over a rect); the skew-order '
              'semantics are unresolved - see skewx_before_rotation',
    'GetSkewXBeforeRotation': 'skew-order readback - see skewx_before_rotation',
    'GetSkewYBeforeRotation': 'skew-order readback - see skewx_before_rotation',
    # spline / fit position + zoom setters whose target geometry needs the
    # actor's natural size the sim does not carry.
    'position': 'SetPosition - spline path time, not an x/y write '
                '(Actor.h:494); no spline model',
    'scaletocover': 'ScaleToCover(rect) - fit-covering scale from natural '
                    'size (Actor.h:553); size math not carried',
    'scaletofit': 'ScaleToFitInside(rect) - fit-inside scale from natural '
                  'size (Actor.h:554); size math not carried',
    # per-edge diffuse + fade families: gradient tinting the color model
    # (single diffuse color + alpha) does not represent.
    'diffuseupperleft': 'per-corner diffuse gradient - flat color model only',
    'diffuseupperright': 'per-corner diffuse gradient - flat color model only',
    'diffuselowerleft': 'per-corner diffuse gradient - flat color model only',
    'diffuselowerright': 'per-corner diffuse gradient - flat color model only',
    'diffuseleftedge': 'per-edge diffuse gradient - flat color model only',
    'diffuserightedge': 'per-edge diffuse gradient - flat color model only',
    'diffusetopedge': 'per-edge diffuse gradient - flat color model only',
    'diffusebottomedge': 'per-edge diffuse gradient - flat color model only',
    'getdiffuse': 'diffuse color readback - flat color model, no consumer',
    'glow': 'additive glow color overlay - no glow compositing pass yet',
    'fade': 'edge-fade all sides - fade gradients not composited',
    'fadeleft': 'left edge fade - fade gradients not composited',
    'faderight': 'right edge fade - fade gradients not composited',
    'fadetop': 'top edge fade - fade gradients not composited',
    'fadebottom': 'bottom edge fade - fade gradients not composited',
    'fadeh': 'horizontal edge fade - fade gradients not composited',
    'fadev': 'vertical edge fade - fade gradients not composited',
    # primitives + live tiers already deferred in lua_api.
    'luaeffect': 'arbitrary per-frame Lua effect (SetEffectLua) - live '
                 'channel tier, chart Lua owns it',
    'tween': 'tween with a custom Lua easing function - live channel tier',
    'floorwag': 'fork wag variant (Effect not in openitg; Actor.clean.c '
                'apply-math is COMDAT-folded so its offset/floor behavior '
                'cannot be pinned) - recorded as an oscillator span, '
                'synthesis left deferred until a source fixes it',
    'rainbow': 'color oscillator - recorded as a span; color-oscillator '
               'synthesis pending',
    'diffuseshift': 'color oscillator - see rainbow',
    'diffuseblink': 'color oscillator - see rainbow',
    'diffuseramp': 'color oscillator - see rainbow',
    'glowshift': 'glow oscillator - see rainbow',
    'glowblink': 'glow oscillator - see rainbow',
    'glowramp': 'glow oscillator - see rainbow',
    'effectoffset': 'effect phase offset - oscillator param, recorded but '
                    'the position/rotation synthesis does not read it yet',
    # effect readback getters: the effect draw-time contribution the sim
    # deliberately excludes from GetX (Actor.cpp:248 - effects apply to the
    # draw temp, not m_current), so there is nothing to read back yet.
    'GetEffectDelta': 'effect phase delta readback - effect draw-time state '
                      'the sim excludes from reads (Actor.cpp:248)',
    'GetEffectX': 'effect x contribution readback - see GetEffectDelta',
    'GetEffectY': 'effect y contribution readback - see GetEffectDelta',
    'GetEffectZ': 'effect z contribution readback - see GetEffectDelta',
    'GetEffectRotationX': 'effect rotation readback - see GetEffectDelta',
    'GetEffectRotationY': 'effect rotation readback - see GetEffectDelta',
    'GetEffectRotationZ': 'effect rotation readback - see GetEffectDelta',
    # shader per-actor binds - the GL executor owns these.
    'SetShader': 'per-actor shader program bind - GL executor',
    'GetShader': 'per-actor shader program handle - GL executor',
    'ClearShader': 'per-actor shader clear - GL executor',
    # meta / tree readback returning sizes the sim does not carry.
    'GetWidth': 'natural width - the sim does not carry actor pixel size',
    'GetHeight': 'natural height - see GetWidth',
    'SetWidth': 'overrides natural width - size math not carried',
    'SetHeight': 'overrides natural height - size math not carried',
    'GetParent': 'actor-tree parent - env registry owns tree navigation',
}


# -- mechanisms 5/7/8/10/11: verbs SimActor handles by name ------------------
# These are NOT tables the dispatch keys off - SimActor.poke matches them
# directly (tween queue, effect spans, immediate bits, texture/proxy binds,
# command queue). Every name here is in actor_api_names.txt; the value is
# the owning mechanism tag, so the completeness test can classify each
# registered name without a table lookup it will not perform at runtime.
HANDLED_BY_NAME: dict = {
    # mechanism 5: tween-queue ops
    'linear': 'tween-queue', 'accelerate': 'tween-queue',
    'decelerate': 'tween-queue', 'spring': 'tween-queue',
    'bouncebegin': 'tween-queue', 'bounceend': 'tween-queue',
    'sleep': 'tween-queue', 'stoptweening': 'tween-queue',
    'finishtweening': 'tween-queue', 'hurrytweening': 'tween-queue',
    # mechanism 7: effect span kinds + params
    'vibrate': 'effect-span', 'wag': 'effect-span', 'bob': 'effect-span',
    'bounce': 'effect-span', 'spin': 'effect-span', 'stopeffect': 'effect-span',
    # zoom oscillators: recorded as effect spans, synthesized into a
    # scale_x/scale_y keyframe stream (mirrors the position oscillators).
    'pulse': 'effect-span', 'pulseramp': 'effect-span',
    'effectmagnitude': 'effect-span', 'effectperiod': 'effect-span',
    'effectclock': 'effect-span', 'effectdelay': 'effect-span',
    'effectcolor1': 'effect-span', 'effectcolor2': 'effect-span',
    'GetSecsIntoEffect': 'effect-span', 'GetTweenTimeLeft': 'effect-span',
    # StretchTo(rect): position at rect center + absolute size to fill it
    # (SimActor._poke_multi_arg does the center/extent math).
    'stretchto': 'size-fill',
    # mechanism 8: immediate bit / hint writes SimActor models
    'hidden': 'visibility', 'visible': 'visibility',
    'blend': 'blend', 'additiveblend': 'blend',
    'setstate': 'sprite', 'animate': 'sprite', 'play': 'sprite',
    'pause': 'sprite',
    'diffuse': 'diffuse', 'diffusecolor': 'diffuse',
    # modchart scratch state (SimActor handles aux/addaux directly, with
    # getaux as the current-read getter above)
    'aux': 'scratch', 'addaux': 'scratch',
    # mechanism 11: command / message registry
    'playcommand': 'command', 'queuecommand': 'command',
    'queuemessage': 'command', 'addcommand': 'command',
    'removecommand': 'command', 'hascommand': 'command', 'cmd': 'command',
}


def all_targets() -> dict:
    """name -> (mechanism, target) for every verb this module maps, so the
    completeness test can assert coverage. `target` is the property / tuple
    a write or read drives, the reason for IGNORED/DEFERRED, or the
    mechanism tag for the handled-by-name verbs."""
    out: dict = {}
    for name, prop in SCALAR_SETTERS.items():
        out[name] = ('scalar-setter', prop)
    for name, prop in ADD_SETTERS.items():
        out[name] = ('add-setter', prop)
    for name, props in BULK_SETTERS.items():
        out[name] = ('bulk-setter', props)
    for name, props in BULK_ADD_SETTERS.items():
        out[name] = ('bulk-add-setter', props)
    for name, props in CROP_COMPOSITES.items():
        out[name] = ('crop-composite', props)
    for name, props in SIZE_PAIR_SETTERS.items():
        out[name] = ('size-setter', props)
    for name, prop in SIZE_AXIS_SETTERS.items():
        out[name] = ('size-setter', prop)
    for name, spec in GETTERS.items():
        out[name] = ('getter', spec)
    for name, spec in TUPLE_GETTERS.items():
        out[name] = ('tuple-getter', spec)
    for name, tag in HANDLED_BY_NAME.items():
        out.setdefault(name, ('handled', tag))
    for name, reason in IGNORED.items():
        out[name] = ('ignored', reason)
    for name, reason in DEFERRED.items():
        out[name] = ('deferred', reason)
    return out
