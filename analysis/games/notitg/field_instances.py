"""Field-instance consumer: NotITG playfield instances as
EffectFrame.fields.

A NotITG frame draws playfield captures: the two player fields (real
tournament players, each an always-rendered independently-modded group),
ActorProxy copies re-rendering a player's notefield, and
ActorFrameTexture copies replaying the previous frame's composed screen.
The compile step emits each as a generic instance carrying ONE composed
transform channel (games/notitg/field_compose); this effect samples
`instance['transform'].at(t)`, conjugates the result by the chart-region
design map, and assigns the capture scope. It holds no per-property, no
per-player, and no per-chart knowledge.

The renderer's field pipeline (effects/base.py,
qt_renderer._blit_field_instances) renders the field layer group once
into a window-sized pixmap (twice for dual-player charts - the second
capture samples player 2's mod channels, see `SecondFieldSpec`), then
blits it once per `(transform, opacity, scope)` in SCREEN space, clipped
to the chart region. `(None, 1.0, 'field')` is the untouched original.

Scopes: proxy/player instances blit a notefield capture - player 2's
instances the independently-modded second capture ('field2') when a
`second_field` spec is present, everything else the primary ('field').
AFT instances blit the chart-area capture their source node took at its
draw position (backdrop + field blits, before any post-node sampler):
'screen' samplers draw AFTER the node, so they show this frame's fresh
capture (identity is a no-op re-draw, a transform is a screen-copy
toss); 'screen_prev' samplers draw BEFORE it, so they show the previous
frame's (their own blit lands in the next capture - the feedback leg
that accumulates echo trails).

AFT chains (games/notitg/aft_chains): a 2-stage chain node captures ONE
isolated upstream AFT (a post-processed field copy), not the whole
screen. Its consumers carry `capture_source` (the upstream node name)
and key their freeze on it, so all consumers of that chain node share
its retained isolated capture. Absent a chain (gat 1, single AFT), the
key is the sampler's own name and the capture is the whole screen -
byte-identical. The render loop snapshotting the isolated content under
that key (a sub-composite, not the finished frame) is DEFERRED - it
needs the GL executor's named render targets, out of this consumer's
compile-time reach.

Design-space mapping: a sampled homography is authored in SM's 640x480
screen space; the screen transform is the conjugation M . H . M^-1
where M maps SM design space onto the chart region (kept in lockstep
with the storyboard renderer's _design_transform)."""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import QTransform

from analysis.player.render import transform3d
from analysis.player.render.effects.base import EffectFrame

_DESIGN_W = 640.0
_DESIGN_H = 480.0

# The primary (player-1) field capture. Player-N proxies blit from
# per-player captures `field{N}` (field2, field3, ...), each an
# independently-modded re-render of that player's notefield - a chart
# can enable up to 8 players and proxy any of them (the SRT charts'
# decorative field copies).
_PROXY_SCOPE = 'field'


def _player_scope(number) -> str:
    """The capture-slot scope for player `number`: the primary 'field'
    for player 1, else 'field{N}' (its own per-player re-render)."""
    return _PROXY_SCOPE if number <= 1 else f'field{number}'


_AFT_SCOPE = 'screen'
# Pre-node AFT samplers: drawn before their source node captures, so
# they show the previous frame's capture (the feedback leg of trails).
_AFT_PREV_SCOPE = 'screen_prev'
# An AFT-rig curtain quad blitted at its tree position (covers the
# proxies under it; samplers above it stay visible).
_FILL_SCOPE = 'fill'

_IDENTITY_H = np.eye(3)


class PlayerFieldsSpec:
    """Instruction to the renderer to render the field layers once PER
    non-primary player, each into its own capture slot.

    `note_mods` maps a 1-based player number (>= 2) to that player's
    mod consumer. For each, the renderer re-renders the field with that
    consumer swapped in (so its notes, receptor offsets, and reverse
    baseline evaluate against that player's channels - a chart mods each
    player independently) into slot `field{N}`; a proxy of player N
    blits from it. Everything but the sampled (mod, player) channels is
    identical across the captures. Player 1 is the primary 'field'
    capture, always rendered, so it is not in this map."""

    def __init__(self, note_mods):
        self.note_mods = dict(note_mods)


def _design_map(chart_rect):
    """(kx, ky, ox, oy) mapping SM 640x480 design space onto the chart
    region by stretching each axis to fill - the engine's own policy:
    NotITG renders a fixed 640x480 design screen and stretches it to the
    window, so widescreen play widens the content instead of
    letterboxing it. Kept in lockstep with the notitg storyboard's
    _design_transform ('stretch' fit) so a field copy lines up with the
    storyboard actors drawn over it."""
    x, y, w, h = chart_rect
    return w / _DESIGN_W, h / _DESIGN_H, x, y


def design_box(chart_rect) -> QRectF:
    """The mapped design box in screen space (the hard crop region),
    matching _design_map: under the stretch fit it IS the chart region -
    the engine clips at its render-target edge."""
    return QRectF(*chart_rect)


def _screen_transform(H, kx, ky, ox, oy) -> QTransform | None:
    """Window-space QTransform for a sampled capture->design homography:
    None for the identity (the untransformed-blit fast path), else the
    design-map conjugation M^-1 . H . M (Qt row-vector composition:
    the leftmost factor applies to a point first)."""
    if np.allclose(H, _IDENTITY_H, atol=1e-9):
        return None
    to_design = QTransform(1.0 / kx, 0.0, 0.0, 1.0 / ky, -ox / kx, -oy / ky)
    to_screen = QTransform(kx, 0.0, 0.0, ky, ox, oy)
    return to_design * transform3d.qtransform_from_h(H) * to_screen


class NotitgFieldInstances:
    """Effect turning compiled field instances into per-frame
    `EffectFrame.fields`.

    `base_hidden` samples 1.0 while the chart hides the real NoteField
    (`P1:hidden(1)`) to let copies stand in. Single-player: the identity
    original draws unless base-hidden, alongside any live copies (no
    instances live and base visible -> None, the renderer's direct-draw
    fast path). Dual-player (`second_field` present): the two player
    instances ARE the originals - base-hidden drops them while copies
    own the field - and the frame always takes the capture path (a
    zero-opacity placeholder keeps `fields` non-empty) so the second
    capture renders."""

    def __init__(self, instances, base_hidden=None, player_fields=None):
        # `instances` is a fixed sequence (eager) OR a PROVIDER callable
        # returning the current instance list (LAZY: the set grows as proxy/AFT
        # bindings fire during playback - the consumer re-reads it every frame
        # with no cross-frame state, so a growing set just lights up copies as
        # they appear).
        self._provider = instances if callable(instances) else None
        self._instances = None if self._provider else tuple(instances)
        self._base_hidden = base_hidden
        self._player_fields_spec = player_fields
        self._player_fields = (player_fields.note_mods
                               if player_fields is not None else {})

    def _current_instances(self):
        return self._provider() if self._provider is not None else self._instances

    def __bool__(self):
        # A lazy provider may be empty now but grow later; keep the effect alive.
        return True if self._provider is not None else bool(self._instances)

    def at(self, ctx) -> EffectFrame | None:
        instances = self._current_instances()
        if not instances:
            return None
        t = float(ctx.t_now)
        base_hidden = self._base_field_hidden(t)
        kx, ky, ox, oy = _design_map(ctx.chart_rect)

        entries = []
        for inst in instances:
            if inst['kind'] == 'player' and base_hidden:
                continue
            sampled = inst['transform'].at(t)
            if sampled is None:
                continue
            H, alpha = sampled
            entries.append((_screen_transform(H, kx, ky, ox, oy),
                            min(1.0, alpha), self._scope(inst),
                            self._extra(inst, t)))

        if self._player_fields_spec is not None:
            return EffectFrame(
                fields=tuple(entries or [(None, 0.0, _PROXY_SCOPE)]),
                second_field=self._player_fields_spec)
        return self._single_frame(base_hidden, entries)

    def _single_frame(self, base_hidden, entries) -> EffectFrame | None:
        """Single-player: one centered identity original, present only
        alongside copies (or a placeholder when the base is hidden).
        Base visible with no copies -> None (the renderer's fast path
        draws the base directly)."""
        if not base_hidden and not entries:
            return None
        if base_hidden:
            # Base suppressed (copies replace it). A zero-opacity
            # placeholder keeps `fields` non-empty even with no visible
            # copies, so the renderer takes the capture path.
            entries = entries or [(None, 0.0, _PROXY_SCOPE)]
        else:
            entries = [(None, 1.0, _PROXY_SCOPE), *entries]
        return EffectFrame(fields=tuple(entries))

    def _base_field_hidden(self, t) -> bool:
        return (self._base_hidden is not None
                and self._base_hidden.sample(t)[0] >= 0.5)

    def _scope(self, inst) -> str:
        if inst['kind'] == 'fill':
            return _FILL_SCOPE
        if inst['kind'] == 'aft':
            return (_AFT_PREV_SCOPE if inst.get('aft_order') == 'pre'
                    else _AFT_SCOPE)
        # A proxy/player blits from its target player's capture: the
        # primary 'field' for player 1, or the per-player re-render
        # 'field{N}' the spec provides for player N > 1.
        player = inst.get('player') or 1
        if player > 1 and player in self._player_fields:
            return _player_scope(player)
        return _PROXY_SCOPE

    @staticmethod
    def _extra(inst, t):
        """The scope's per-frame payload: a fill's sampled rgb, or an
        aft sampler's (source name, capture-live?) freeze key. None for
        proxy/player blits.

        The freeze key names the ISOLATED render target the sampler
        blits: a 2-stage chain node's `capture_source` (the upstream AFT
        it captured alone) when present, else the sampler's own name (the
        whole-screen capture, the gat 1 path). Keying chain consumers on
        the shared upstream node lets them all blit one retained isolated
        capture - the render-loop side that snapshots that isolated
        content under this key is the deferred piece (see module note)."""
        match inst['kind']:
            case 'fill':
                color = inst.get('color')
                return tuple(color.sample(t)) if color is not None \
                    else (1.0, 1.0, 1.0)
            case 'aft':
                live = inst.get('aft_live')
                key = inst.get('capture_source') or inst['name']
                return (key, live is None or live.sample(t)[0] >= 0.5)
        return None


class NotitgScreenCamera:
    """Whole-scene camera from the per-frame update's top-screen pokes.

    gat_updateproxies drives `SCREENMAN:GetTopScreen()` as a screen-zoom
    camera: it scales the top screen and offsets it by `(1-zoom)*center`
    so the zoom pivots on the design centre (the t=42 pull-back and the
    t=42/t=383 push-in). Those pokes are compiled to a screen-transform
    timeline (x/y/scale in 640x480 design space); this effect maps them
    onto the chart region and emits them on the scene_transform channel,
    the same whole-scene slot as the fluXis camera.

    The design-space transform SM applies is scale-about-origin then
    translate (`zoom(z); x((1-z)*cx); y((1-z)*cy)`), i.e. scale `z` about
    the design centre; we conjugate it by the design map so it pivots on
    the mapped centre in screen space."""

    def __init__(self, timelines):
        self._tl = timelines

    def __bool__(self):
        return self._tl is not None

    def at(self, ctx) -> EffectFrame | None:
        if self._tl is None:
            return None
        t = float(ctx.t_now)
        sx = self._tl['scale_x'].sample(t)[0]
        sy = self._tl['scale_y'].sample(t)[0]
        tx = self._tl['x'].sample(t)[0]
        ty = self._tl['y'].sample(t)[0]
        if sx == 1.0 and sy == 1.0 and tx == 0.0 and ty == 0.0:
            return None
        kx, ky, ox, oy = _design_map(ctx.chart_rect)
        transform = QTransform()
        transform.translate(ox, oy)
        transform.scale(kx, ky)
        transform.translate(tx, ty)
        transform.scale(sx, sy)
        transform.scale(1.0 / kx, 1.0 / ky)
        transform.translate(-ox, -oy)
        return EffectFrame(scene_transform=transform)
