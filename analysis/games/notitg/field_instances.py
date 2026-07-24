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
blits it once per `(transform, opacity, scope, extra, crop)` in SCREEN
space, clipped to the chart region. `(None, 1.0, 'field')` is the
untouched original; `crop` is the instance's sampled SetCrop* insets
(None at rest), a source-space clip on its blit.

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
# A chain-involved AFT node: the renderer snapshots the in-progress
# composite into the node's named slot at this entry's position (the
# engine captures at the node's draw position). Slots retain across
# frames - a hidden node's entry vanishes and the slot freezes, and a
# feedback node's earlier-drawn samplers read last frame's content.
_CAPTURE_SCOPE = 'capture'

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
    capture, always rendered, so it is not in this map.

    `factory` (player number -> consumer) serves the LAZY topology: a
    proxy of player N can bind long after the spec was built (the
    instance provider grows as the chart plays), so the effect calls
    `ensure` with the players its current instances reference and the
    spec mints the missing consumers on sight."""

    def __init__(self, note_mods, factory=None):
        self.note_mods = dict(note_mods)
        self._factory = factory

    def ensure(self, players) -> None:
        """Mint consumers for any of `players` (1-based, > 1) not in the
        map yet. No-op without a factory (the eager spec is complete)."""
        if self._factory is None:
            return
        for number in players:
            if number not in self.note_mods:
                self.note_mods[number] = self._factory(number)


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


def _fold_stage_chain(stages, slots, inst, H, alpha, crop, extra):
    """Fold an aft consumer's stage chain into its blit.

    Pure-transform stages never materialize: every consumer of the same
    chain blits the ROOT node's snapshot slot under the composed
    homography (render-once/consume-many - textures are immutable, so
    sharing is free; only an effect-baking stage would fork a copy).
    Walks the consumer's direct source through the sampled `stages`
    (row-vector composition: content passes the innermost stage first),
    multiplying alphas. The innermost stage's crop survives as the
    source clip; the consumer's own crop and outer stage crops are
    mid-chain clips a single blit cannot express - dropped here, and
    restored only where a stage materializes (the shader tier).

    Returns the folded `(H, alpha, crop, extra)`: extra keys the ROOT
    slot when the walk lands on a materialized snapshot node, else the
    legacy (capture_source-or-name, live) pair untouched - the gat 1
    whole-screen path, byte-identical.

    A consumer whose DIRECT source is a stage node with no stage record
    this frame (its captured sprite is hidden - the isolation premise
    is not currently observable) serves that node's own at-position
    slot instead: the node is still live and capturing the whole screen
    at its position, so the stale chain-root snapshot would be wrong
    (gat 2's monitor sampler through the chickenstrips ending)."""
    node = inst.get('aft_node')
    folded = False
    if node not in stages and node in slots and extra is not None:
        return H, alpha, crop, (node,) + tuple(extra[1:])
    while node in stages:
        H_s, alpha_s, crop_s, source = stages[node]
        H = H_s @ H
        alpha *= alpha_s
        crop = crop_s
        node = source
        folded = True
    if (node in slots or folded) and extra is not None:
        # Key the ROOT so the renderer serves its slot (when absent this
        # frame, the retained slot from the node's last live frame).
        # The tail carries the live flag and any shaded-blit payload
        # through unchanged.
        return H, alpha, crop, (node,) + tuple(extra[1:])
    return H, alpha, crop, extra


def _z_sort_entries(entries, sort_keys) -> None:
    """Reorder entries within each SetDrawByZPosition group by their
    sampled z, ascending with original order breaking ties (the engine's
    stable sort - ActorUtil::SortByZPosition), in place. Group members
    re-slot into the group's own index positions, so entries outside
    the group keep their exact draw positions."""
    groups = {}
    for i, key in enumerate(sort_keys):
        if key is not None:
            groups.setdefault(key[0], []).append(i)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        ranked = sorted((sort_keys[i][1], i) for i in indices)
        originals = {i: entries[i] for i in indices}
        for slot, (_z, src) in zip(indices, ranked):
            entries[slot] = originals[src]


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
        if self._player_fields_spec is not None:
            # Late-binding players (lazy provider growth): consumers for
            # every player the current copies reference must exist BEFORE
            # scopes resolve, or a fresh proxy blits player 1's capture.
            self._player_fields_spec.ensure(
                {player for inst in instances
                 if inst['kind'] in ('proxy', 'player')
                 and (player := inst.get('player') or 1) > 1})
        t = float(ctx.t_now)
        base_hidden = self._base_field_hidden(t)
        kx, ky, ox, oy = _design_map(ctx.chart_rect)

        entries = []
        sort_keys = []
        stages = {}
        slots = set()
        for inst in instances:
            kind = inst['kind']
            if kind == 'stage':
                # An isolating AFT node: record its captured sprite's
                # sampled transform for the fold below; never a draw of
                # its own. Hidden this frame -> absent -> consumers fall
                # to the legacy screen path (frozen-content chains are a
                # renderer-retention concern, not a transform one).
                sampled = inst['transform'].at(t)
                if sampled is not None:
                    stages[inst['name']] = (
                        *sampled, inst['transform'].crop_at(t),
                        inst['source'])
                continue
            if kind == 'player' and base_hidden:
                continue
            sampled = inst['transform'].at(t)
            if sampled is None:
                continue
            H, alpha = sampled
            extra = self._extra(inst, t)
            crop = inst['transform'].crop_at(t)
            if kind == 'capture':
                slots.add(inst['name'])
            elif kind == 'aft':
                H, alpha, crop, extra = _fold_stage_chain(
                    stages, slots, inst, H, alpha, crop, extra)
            entries.append((_screen_transform(H, kx, ky, ox, oy),
                            min(1.0, alpha), self._scope(inst),
                            extra, crop))
            group = inst.get('z_group')
            sort_keys.append(
                None if group is None
                else (group, inst['z_sort'].sample(t)[0]))
        _z_sort_entries(entries, sort_keys)

        spec = self._player_fields_spec
        if spec is not None and spec.note_mods:
            return EffectFrame(
                fields=tuple(entries or [(None, 0.0, _PROXY_SCOPE)]),
                second_field=spec)
        # A lazy factory spec with no minted consumer yet is inert: the
        # chart has referenced no player > 1, so the single-player frame
        # (and its direct-draw fast path) stands.
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
        if inst['kind'] == 'capture':
            return _CAPTURE_SCOPE
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
                # Freeze/slot key: the isolated upstream for chain
                # consumers, else the sampler's SOURCE NODE (its slot /
                # preserve-texture identity - the engine freezes per
                # node, not per sampler), else the sampler name (a
                # source outside the node set, the last-resort key).
                key = (inst.get('capture_source') or inst.get('aft_node')
                       or inst['name'])
                live_now = live is None or live.sample(t)[0] >= 0.5
                frag = inst.get('frag')
                mesh = inst.get('mesh')
                color = inst.get('color')
                blend = inst.get('blend_add')
                tint = tuple(color.sample(t)) if color is not None \
                    else (1.0, 1.0, 1.0)
                additive = blend is not None and blend.sample(t)[0] >= 0.5
                if frag is None and mesh is None and not additive \
                        and tint == (1.0, 1.0, 1.0):
                    return (key, live_now)
                # The blit-style payload: the sampler's .frag path, its
                # uniform pokes sampled now, the diffuse rgb tint, the
                # additive-blend flag (`blend('add')` - black
                # contributes nothing, overlaps sum; opaque source-over
                # instead turns dark copies into occluders, which tiled
                # the cyriak recursion into triangle holes), and the
                # frag's uniformTexture file binds.
                uniforms = {name: tl.sample(t)[0] for name, tl in
                            (inst.get('frag_uniforms') or {}).items()}
                return (key, live_now, (frag, uniforms, tint, additive,
                                        inst.get('frag_samplers') or None),
                        mesh)
            case 'capture':
                # The slot name the renderer snapshots the in-progress
                # composite into at this entry's position.
                return inst['name']
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

    def __init__(self, timelines, oscillator=None):
        self._tl = timelines
        # A `.channels` holder ({prop: sampleable} or None): the screen's
        # effect-oscillator jitter delta (`screen:vibrate()` +
        # per-frame effectmagnitude - the datamosh scene shake), summed
        # onto the camera translation. A holder, not a dict, so the
        # lazy path's background sweep can fill it in after open.
        self._osc = oscillator

    def __bool__(self):
        return self._tl is not None or self._osc is not None

    def at(self, ctx) -> EffectFrame | None:
        t = float(ctx.t_now)
        tl = self._tl
        sx = tl['scale_x'].sample(t)[0] if tl else 1.0
        sy = tl['scale_y'].sample(t)[0] if tl else 1.0
        tx = tl['x'].sample(t)[0] if tl else 0.0
        ty = tl['y'].sample(t)[0] if tl else 0.0
        shake = getattr(self._osc, 'channels', None)
        if shake:
            x_delta = shake.get('x')
            y_delta = shake.get('y')
            tx += x_delta.sample(t)[0] if x_delta is not None else 0.0
            ty += y_delta.sample(t)[0] if y_delta is not None else 0.0
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
