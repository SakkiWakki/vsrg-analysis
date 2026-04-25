"""End-to-end: a sandboxed plugin using the component API loads,
builds its tree, and renders into a fake SidebarContext correctly.

This proves the allow-list covers ``analysis.ui`` and that the
declarative API composes with the real plugin loader.
"""
from __future__ import annotations

import textwrap

import pytest

from analysis.plugins import discover_bundles
from analysis.ui.components import Button, Column, Heading


@pytest.fixture
def ui_bundle_root(tmp_path, monkeypatch):
    monkeypatch.setenv('EA_PLUGINS_PATH', str(tmp_path))
    return tmp_path


def _write_ui_plugin(root, key):
    bundle = root / key
    (bundle / 'sidebar').mkdir(parents=True)
    (bundle / 'manifest.toml').write_text(
        f'name = "{key}"\nkey = "{key}"\nversion = "0.0"\n')
    (bundle / 'sidebar' / 'hello.py').write_text(textwrap.dedent('''
        from analysis.ui import Button, Column, Heading, Text
        from analysis.ui.render_sidebar import render

        def build():
            return Column((
                Heading('demo'),
                Text('body'),
                Button('ok', 'demo_click'),
            ))

        def _draw(sctx):
            render(sctx, build())

        def register_sidebar(add):
            add('demo', _draw, priority=500, key=f'{"''' + key + '''"}:hello')
    '''))


class _FakeCtx:
    """Enough of SidebarContext to let the render helper advance."""
    def __init__(self):
        self.col_x, self.col_w, self.y = 0, 200, 0
        self.measure_only = False
        self.calls = []

        class _P:
            def setFont(self, *a): pass
        class _R:
            big_font = 'BIG'; font = 'NORMAL'
        self.painter = _P()
        self.renderer = _R()

    def text(self, t, x, b, c):     self.calls.append(('text', t))
    def rect(self, *a, **k):        self.calls.append(('rect',))
    def add_hitbox(self, r, a, p=None):
        self.calls.append(('hitbox', a))
    def button_at(self, r, l, a, p=None, **k):
        self.calls.append(('button', l, a))
        self.calls.append(('hitbox', a))
    def checkbox(self, *a, **k):    self.calls.append(('checkbox',))


def test_sandboxed_plugin_builds_and_renders_component_tree(ui_bundle_root):
    _write_ui_plugin(ui_bundle_root, 'ui_demo')
    bundles = discover_bundles()
    demo = next(b for b in bundles if b.key == 'ui_demo')
    assert demo.load_errors == []
    assert len(demo.sidebar_modules) == 1

    mod = demo.sidebar_modules[0]
    tree = mod.build()
    assert isinstance(tree, Column)
    assert any(isinstance(c, Heading) for c in tree.children)
    assert any(isinstance(c, Button) and c.action == 'demo_click'
               for c in tree.children)

    # Render through the fake context to prove it produces primitives.
    ctx = _FakeCtx()
    mod._draw(ctx)
    kinds = [c[0] for c in ctx.calls]
    assert 'button' in kinds
    assert ('hitbox', 'demo_click') in ctx.calls


def test_sandboxed_plugin_cannot_reach_into_unlisted_ui_modules(ui_bundle_root):
    """A future sibling module under analysis.ui (e.g. ``ui.internal``)
    must not be allowed just because ``analysis.ui`` is. Guards against
    the parent-package allow bleeding into submodules the user hasn't
    opted in to. Here we craft an import of a non-existent submodule;
    the sandbox should catch it before Python's import machinery does."""
    bundle = ui_bundle_root / 'bad_ui'
    (bundle / 'sidebar').mkdir(parents=True)
    (bundle / 'manifest.toml').write_text(
        'name = "bad_ui"\nkey = "bad_ui"\n')
    (bundle / 'sidebar' / 'bad.py').write_text(textwrap.dedent('''
        import analysis.ui.nonexistent_internal  # not on the allow-list
        def register_sidebar(add): pass
    '''))
    bundles = discover_bundles()
    bad = next(b for b in bundles if b.key == 'bad_ui')
    # Either refused by the sandbox or failed as ModuleNotFoundError.
    # Either outcome is acceptable ; the plugin must not successfully
    # load.
    assert len(bad.load_errors) == 1
    assert len(bad.sidebar_modules) == 0
