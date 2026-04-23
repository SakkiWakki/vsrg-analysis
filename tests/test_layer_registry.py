from __future__ import annotations

from analysis.components import (
    LayerDeclaration,
    LayerPlacement,
    LAYER_AFTER,
    LAYER_GROUP,
    LAYER_INSIDE,
    Manifest,
    SURFACE_GUI,
)
from analysis.config.store import ConfigStore
from analysis.player.render.layer_registry import LayerRegistry


def _store(tmp_path):
    store = ConfigStore(tmp_path / 'config.json', autosave=False)
    store.load()
    return store


def _manifest(key, *layers):
    return Manifest(
        key=key,
        name=key,
        supported_surfaces={SURFACE_GUI},
        layers=layers,
    )


def test_builtin_plan_keeps_hud_after_free_sections(tmp_path):
    registry = LayerRegistry(config=_store(tmp_path))
    plan = registry.render_plan({})
    names = [name for name, _, _ in plan]
    assert names[-2:] == ['free_sections', 'hud']


def test_plugin_layer_inserts_between_builtin_layers(tmp_path):
    registry = LayerRegistry(config=_store(tmp_path))
    registry.register_manifest(_manifest(
        'bundle:plugin',
        LayerDeclaration(
            key='bundle:after_notes',
            name='After notes',
            placement=LayerPlacement(LAYER_AFTER, 'notes'),
        ),
    ))
    plan = registry.render_plan({})
    names = [name for name, _, _ in plan]
    notes_index = names.index('notes')
    assert names[notes_index + 1] == 'bundle:after_notes'
    assert names[notes_index + 2] == 'chart_extras'


def test_parent_visibility_hides_children(tmp_path):
    registry = LayerRegistry(config=_store(tmp_path))
    assert registry.set_visible('playfield', False) is True
    assert registry.layer_visible('notes') is False
    assert registry.layer_visible('hud') is True


def test_hud_layer_cannot_be_hidden(tmp_path):
    registry = LayerRegistry(config=_store(tmp_path))
    assert registry.set_visible('hud', False) is False
    assert registry.layer_visible('hud') is True


def test_inside_requires_parent_handshake_for_plugin_groups(tmp_path):
    store = _store(tmp_path)
    registry = LayerRegistry(config=store)
    registry.register_manifest(_manifest(
        'bundle:parent',
        LayerDeclaration(
            key='bundle:group',
            name='Group',
            placement=LayerPlacement(LAYER_AFTER, 'notes'),
            kind=LAYER_GROUP,
        ),
    ))
    registry.register_manifest(_manifest(
        'bundle:child',
        LayerDeclaration(
            key='bundle:child_layer',
            name='Child',
            placement=LayerPlacement(LAYER_INSIDE, 'bundle:group'),
        ),
    ))
    failure = registry.failure('bundle:child_layer')
    assert failure is not None
    assert 'does not accept child' in failure.reason


def test_inside_mounts_when_parent_accepts_child(tmp_path):
    store = _store(tmp_path)
    registry = LayerRegistry(config=store)
    registry.register_manifest(_manifest(
        'bundle:parent',
        LayerDeclaration(
            key='bundle:group',
            name='Group',
            placement=LayerPlacement(LAYER_AFTER, 'notes'),
            kind=LAYER_GROUP,
            accepts_children={'bundle:child_layer'},
        ),
    ))
    registry.register_manifest(_manifest(
        'bundle:child',
        LayerDeclaration(
            key='bundle:child_layer',
            name='Child',
            placement=LayerPlacement(LAYER_INSIDE, 'bundle:group'),
        ),
    ))
    tree = registry.layer_tree()
    playfield = tree[0]
    group = next(state for state in playfield.children if state.key == 'bundle:group')
    assert [child.key for child in group.children] == ['bundle:child_layer']
    states = registry.listed_layers()
    keys = [state.key for state in states]
    assert 'bundle:child_layer' in keys
