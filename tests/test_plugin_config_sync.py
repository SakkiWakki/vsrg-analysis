"""Cross-window config propagation tests.

The user-visible promise: toggle a plugin in one window's dialog, the
other window's replay/sidebar picks up the change without a restart.
Here we stand up two independent ``PluginManager`` instances backed by
the *same* ``ConfigStore`` and prove a flip in one shows up in the
other.
"""
from __future__ import annotations

import pytest

from analysis.config.store import ConfigStore
from analysis.player.plugin_api import Stage
from analysis.player.plugin_loader import PluginManager


@pytest.fixture
def store(tmp_path):
    s = ConfigStore(tmp_path / 'config.json', autosave=False)
    s.load()
    return s


def _install_plugin(mgr, key='bundle:widget'):
    mgr.add('widget', lambda ctx, stage: None,
            stages=(Stage.POST_FRAME,), key=key, module='bundle/widget')


def _install_section(mgr, key='bundle:panel'):
    mgr.sidebar.add('panel', lambda sctx: None,
                    key=key, module='bundle/panel')


def test_replay_toggle_propagates_between_managers(store):
    """Window A disables a replay plugin; window B should see
    ``enabled=False`` on its matching plugin immediately."""
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_plugin(a, 'shared:widget')
    _install_plugin(b, 'shared:widget')

    a.set_enabled('shared:widget', False)

    assert b.all_plugins()[0].enabled is False
    assert a.all_plugins()[0].enabled is False


def test_replay_reenable_propagates(store):
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_plugin(a, 'shared:widget')
    _install_plugin(b, 'shared:widget')

    a.set_enabled('shared:widget', False)
    assert b.all_plugins()[0].enabled is False
    a.set_enabled('shared:widget', True)
    assert b.all_plugins()[0].enabled is True


def test_sidebar_toggle_propagates(store):
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_section(a, 'shared:panel')
    _install_section(b, 'shared:panel')

    a.sidebar.set_enabled('shared:panel', False)
    assert b.sidebar.all_sections()[0].enabled is False
    assert b.sidebar.top_sections() == []  # filtered out


def test_managers_registered_late_pick_up_current_state(store):
    """A new window opened after the user already disabled something
    should honor that state — the state lives in the store, not in
    per-manager memory."""
    first = PluginManager(config=store)
    _install_plugin(first, 'shared:widget')
    first.set_enabled('shared:widget', False)

    second = PluginManager(config=store)
    _install_plugin(second, 'shared:widget')
    assert second.all_plugins()[0].enabled is False


def test_unrelated_key_changes_dont_affect_other_plugins(store):
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_plugin(a, 'bundle:one')
    _install_plugin(a, 'bundle:two')
    _install_plugin(b, 'bundle:one')
    _install_plugin(b, 'bundle:two')

    a.set_enabled('bundle:one', False)
    b_states = {p.key: p.enabled for p in b.all_plugins()}
    assert b_states == {'bundle:one': False, 'bundle:two': True}


def test_close_stops_further_updates(store):
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_plugin(a, 'shared:widget')
    _install_plugin(b, 'shared:widget')

    b.close()
    a.set_enabled('shared:widget', False)
    # b closed — shouldn't have been notified.
    assert b.all_plugins()[0].enabled is True


def test_runtime_disabled_plugin_stays_off_when_config_flips_on(store):
    """If a plugin crashed during draw and got latched off, a config
    toggle *should* re-enable it (user explicitly asked for it back).
    But a config toggle from *another* window shouldn't silently bring
    back a known-broken plugin on this one."""
    a = PluginManager(config=store)
    b = PluginManager(config=store)
    _install_plugin(a, 'shared:boom')
    _install_plugin(b, 'shared:boom')

    # Simulate a runtime failure on b.
    b.all_plugins()[0].enabled = False
    b._runtime_disabled.add('shared:boom')

    # a flips it off, then back on. b's plugin should stay disabled
    # because b's runtime latch is still set.
    a.set_enabled('shared:boom', False)
    a.set_enabled('shared:boom', True)
    assert b.all_plugins()[0].enabled is False

    # But if b's own dialog toggles it on, the latch clears.
    b.set_enabled('shared:boom', True)
    assert b.all_plugins()[0].enabled is True


def test_config_is_persisted_across_managers(tmp_path):
    """A fresh store reading the same file should produce the same
    disabled set."""
    path = tmp_path / 'config.json'
    store = ConfigStore(path)
    store.load()
    mgr = PluginManager(config=store)
    _install_plugin(mgr, 'persist:widget')
    mgr.set_enabled('persist:widget', False)
    store.flush()

    store2 = ConfigStore(path)
    store2.load()
    mgr2 = PluginManager(config=store2)
    _install_plugin(mgr2, 'persist:widget')
    assert mgr2.all_plugins()[0].enabled is False
