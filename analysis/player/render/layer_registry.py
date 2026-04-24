from __future__ import annotations

from dataclasses import dataclass, field

from analysis.components.api import (
    LAYER_AFTER,
    LAYER_BEFORE,
    LAYER_GROUP,
    LAYER_INSIDE,
    LAYER_LEAF,
    LayerDeclaration,
    LayerPlacement,
    LayerState,
)
from analysis.player.plugin_api import Stage


ROOT_LAYER = 'root'
PLAYFIELD_LAYER = 'playfield'
HUD_GROUP_LAYER = 'hud_group'
FREE_SECTIONS_LAYER = 'free_sections'

_BUILTIN_OWNER = 'builtin'


@dataclass
class Layer:
    key: str
    name: str
    owner: str
    kind: str
    placement: LayerPlacement | None = None
    draw: object | None = None
    stage: Stage | None = None
    default_visible: bool = True
    can_hide: bool = True
    listed: bool = True
    accepts_children: frozenset[str] = field(default_factory=frozenset)
    builtin: bool = False
    parent: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LayerFailure:
    key: str
    owner: str
    reason: str


class LayerRegistry:
    def __init__(self, *, config):
        self._config = config
        self._declared: list[Layer] = []
        # Replay-scoped layers (per-game `NoteType`s). Kept separate so
        # loading a new replay swaps only these without rebuilding the
        # plugin-declared ones.
        self._replay_declared: list[Layer] = []
        self._layers: dict[str, Layer] = {}
        self._failures: dict[str, LayerFailure] = {}
        self._rebuild()

    def register_manifest(self, manifest) -> None:
        for decl in manifest.layers:
            self._declared.append(self._from_declaration(decl, owner=manifest.key))
        self._rebuild()

    def register_components(self, components) -> None:
        for comp in components:
            self.register_manifest(comp.manifest)

    def register_note_types(self, note_types) -> None:
        """Replace the replay-scoped note-type layers with a fresh set.
        Each entry in `note_types` is a `NoteType` from `layers/notes.py`;
        every one becomes a leaf layer under PLAYFIELD_LAYER so the HUD
        can toggle it individually. Calling this again (e.g. on replay
        swap) drops the previous set before attaching the new one."""
        self._replay_declared = [
            Layer(
                key=nt.key,
                name=nt.name,
                owner='note_type',
                kind=LAYER_LEAF,
                draw=nt.draw,
                stage=nt.stage,
                placement=LayerPlacement(relation=LAYER_INSIDE,
                                         target=PLAYFIELD_LAYER),
                listed=True,
            )
            for nt in note_types
        ]
        self._rebuild()

    def clear_note_types(self) -> None:
        self._replay_declared = []
        self._rebuild()

    def render_plan(self, draw_lookup: dict[str, object]) -> tuple[tuple[str, object | None, Stage | None], ...]:
        plan: list[tuple[str, object | None, Stage | None]] = []
        for child in self._layers[ROOT_LAYER].children:
            self._append_plan(child, draw_lookup, plan)
        return tuple(plan)

    def layer_visible(self, key: str) -> bool:
        layer_key = str(key)
        if layer_key not in self._layers:
            return False
        return self._effective_visible(layer_key)

    def layer_tree(self) -> tuple[LayerState, ...]:
        root = self._layers[ROOT_LAYER]
        return tuple(
            self._state_for(child, parent_visible=True, depth=0)
            for child in root.children
        )

    def listed_layers(self) -> tuple[LayerState, ...]:
        out: list[LayerState] = []
        for state in self.layer_tree():
            self._append_listed(state, out)
        return tuple(out)

    def toggle(self, key: str) -> bool:
        layer = self._layers.get(str(key))
        if layer is None or not layer.can_hide:
            return False
        current = self._local_visible(layer)
        return bool(self._config.set(self._path(layer.key), not current))

    def set_visible(self, key: str, visible: bool) -> bool:
        layer = self._layers.get(str(key))
        if layer is None or not layer.can_hide:
            return False
        return bool(self._config.set(self._path(layer.key), bool(visible)))

    def failure(self, key: str) -> LayerFailure | None:
        return self._failures.get(str(key))

    def _append_plan(self, key: str, draw_lookup: dict[str, object], plan: list[tuple[str, object | None, Stage | None]]) -> None:
        layer = self._layers[key]
        draw = layer.draw
        if isinstance(draw, str):
            draw = draw_lookup.get(draw)
        if layer.kind == LAYER_LEAF:
            plan.append((layer.key, draw, layer.stage))
            return
        for child in layer.children:
            self._append_plan(child, draw_lookup, plan)

    def _append_listed(self, state: LayerState, out: list[LayerState]) -> None:
        if state.listed:
            out.append(state)
        for child in state.children:
            self._append_listed(child, out)

    def _path(self, key: str) -> str:
        return f'player.layer_visibility.{key}'

    def _local_visible(self, layer: Layer) -> bool:
        if not layer.can_hide:
            return True
        return bool(self._config.get(self._path(layer.key), layer.default_visible))

    def _effective_visible(self, key: str) -> bool:
        layer = self._layers[key]
        local_visible = self._local_visible(layer)
        if not local_visible:
            return False
        parent = layer.parent
        if parent is None:
            return True
        return self._effective_visible(parent)

    def _state_for(self, key: str, *, parent_visible: bool, depth: int) -> LayerState:
        layer = self._layers[key]
        local_visible = self._local_visible(layer)
        visible = parent_visible and local_visible
        children = tuple(
            self._state_for(child, parent_visible=visible, depth=depth + 1)
            for child in layer.children
        )
        return LayerState(
            key=layer.key,
            name=layer.name,
            kind=layer.kind,
            owner=layer.owner,
            parent=layer.parent,
            depth=depth,
            local_visible=local_visible,
            visible=visible,
            can_hide=layer.can_hide,
            listed=layer.listed,
            children=children,
        )

    def _from_declaration(self, decl: LayerDeclaration, *, owner: str) -> Layer:
        return Layer(
            key=decl.key,
            name=decl.name,
            owner=owner,
            kind=decl.kind,
            placement=decl.placement,
            default_visible=decl.default_visible,
            can_hide=decl.can_hide,
            listed=decl.listed,
            accepts_children=decl.accepts_children,
        )

    def _rebuild(self) -> None:
        layers = self._builtin_layers()
        pending = [self._clone(layer) for layer in
                   (self._declared + self._replay_declared)]
        while pending:
            next_round: list[Layer] = []
            progressed = False
            reasons: dict[str, str] = {}
            for layer in pending:
                reason = self._attach(layers, layer)
                if reason is None:
                    progressed = True
                    continue
                next_round.append(layer)
                reasons[layer.key] = reason
            if not progressed:
                self._failures = {
                    layer.key: LayerFailure(
                        key=layer.key,
                        owner=layer.owner,
                        reason=reasons[layer.key],
                    )
                    for layer in next_round
                }
                self._layers = layers
                break
            pending = next_round
        else:
            self._layers = layers
            self._failures = {}

    def _builtin_layers(self) -> dict[str, Layer]:
        layers: dict[str, Layer] = {}

        def add(layer: Layer) -> None:
            layers[layer.key] = layer
            if layer.parent is not None:
                layers[layer.parent].children.append(layer.key)

        add(Layer(
            key=ROOT_LAYER,
            name='Root',
            owner=_BUILTIN_OWNER,
            kind=LAYER_GROUP,
            can_hide=False,
            listed=False,
            builtin=True,
        ))
        add(Layer(
            key=PLAYFIELD_LAYER,
            name='Playfield',
            owner=_BUILTIN_OWNER,
            kind=LAYER_GROUP,
            parent=ROOT_LAYER,
            listed=False,
            builtin=True,
        ))
        add(Layer(
            key='background',
            name='Background',
            owner=_BUILTIN_OWNER,
            kind=LAYER_LEAF,
            parent=PLAYFIELD_LAYER,
            draw='background',
            listed=True,
            builtin=True,
        ))
        add(Layer(
            key='lanes',
            name='Lanes',
            owner=_BUILTIN_OWNER,
            kind=LAYER_LEAF,
            parent=PLAYFIELD_LAYER,
            draw='lanes',
            stage=Stage.AFTER_LANES,
            listed=True,
            builtin=True,
        ))
        add(Layer(
            key='judgment',
            name='Judgment line',
            owner=_BUILTIN_OWNER,
            kind=LAYER_LEAF,
            parent=PLAYFIELD_LAYER,
            draw='judgment',
            stage=Stage.AFTER_JUDGMENT,
            listed=True,
            builtin=True,
        ))
        # The actual note-type leaves (taps/lns/mines/.../ghost_taps)
        # are registered per-replay via `register_note_types()` so an
        # LN-only game never shows a dead 'taps' toggle. Plugin stages
        # `AFTER_NOTES` / `AFTER_GHOSTS` fire via `_note_type_stage()`
        # in qt_renderer, keyed off NoteType.key.
        add(Layer(
            key=HUD_GROUP_LAYER,
            name='HUD',
            owner=_BUILTIN_OWNER,
            kind=LAYER_GROUP,
            parent=ROOT_LAYER,
            can_hide=False,
            listed=False,
            builtin=True,
        ))
        add(Layer(
            key=FREE_SECTIONS_LAYER,
            name='Free sections',
            owner=_BUILTIN_OWNER,
            kind=LAYER_LEAF,
            parent=HUD_GROUP_LAYER,
            draw='free_sections',
            can_hide=False,
            listed=False,
            builtin=True,
        ))
        add(Layer(
            key='hud',
            name='HUD',
            owner=_BUILTIN_OWNER,
            kind=LAYER_LEAF,
            parent=HUD_GROUP_LAYER,
            draw='hud',
            stage=Stage.HUD,
            can_hide=False,
            listed=False,
            builtin=True,
        ))
        return layers

    def _attach(self, layers: dict[str, Layer], layer: Layer) -> str | None:
        if layer.key in layers:
            return 'layer key already exists'
        placement = layer.placement
        if placement is None:
            return 'layer placement is required'
        target = layers.get(placement.target)
        if target is None:
            return f'unknown target layer: {placement.target}'
        relation = placement.relation
        if relation == LAYER_INSIDE:
            if not self._accepts_child(target, layer):
                return f'target layer does not accept child: {target.key}'
            if target.kind != LAYER_GROUP:
                return f'target layer is not a group: {target.key}'
            layer.parent = target.key
            layers[layer.key] = layer
            target.children.append(layer.key)
            return None
        parent_key = target.parent
        if parent_key is None:
            return f'target layer has no parent: {target.key}'
        siblings = layers[parent_key].children
        index = siblings.index(target.key)
        if relation == LAYER_AFTER:
            index += 1
        elif relation != LAYER_BEFORE:
            return f'unknown layer relation: {relation}'
        layer.parent = parent_key
        layers[layer.key] = layer
        siblings.insert(index, layer.key)
        return None

    def _accepts_child(self, target: Layer, child: Layer) -> bool:
        if target.builtin:
            return True
        if not target.accepts_children:
            return False
        accepted_keys = target.accepts_children
        return child.key in accepted_keys or child.owner in accepted_keys

    @staticmethod
    def _clone(layer: Layer) -> Layer:
        return Layer(
            key=layer.key,
            name=layer.name,
            owner=layer.owner,
            kind=layer.kind,
            placement=layer.placement,
            draw=layer.draw,
            stage=layer.stage,
            default_visible=layer.default_visible,
            can_hide=layer.can_hide,
            listed=layer.listed,
            accepts_children=layer.accepts_children,
            builtin=layer.builtin,
        )
