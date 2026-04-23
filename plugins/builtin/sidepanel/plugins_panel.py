"""Built-in sidebar section: collapsible list of registered draw plugins."""
from __future__ import annotations

from analysis.player.render import theme


_CHECKBOX_INSET_X = 6    # gap between row left edge and checkbox
_CHECKBOX_INSET_Y = 3
_LABEL_INDENT = 22       # gap between row left edge and plugin label
_ROW_PANEL_H = 18        # row height inside the expanded plugin list
_TAIL_GAP = 8            # keep-clear margin before the section below
_NAME_MAX_CHARS = 22


def _shorten(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 1)] + '~'


_COLOR_BUNDLE_TRUSTED = (180, 220, 190)
_COLOR_BUNDLE_SANDBOXED = (200, 200, 150)
_COLOR_BUNDLE_REFUSED = (220, 140, 140)


def _draw_plugins_panel(sctx):
    p = sctx.player
    mgr = sctx.renderer.plugins
    plugins = mgr.all_plugins()
    enabled = mgr.enabled_count()
    total = len(plugins)
    bundles = list(getattr(mgr, 'bundles', []))

    sctx.spacer()
    sctx.draw_button(
        f'{"[-]" if p.hud.plugin_panel_open else "[+]"} '
        f'Plugins {enabled}/{total}',
        'toggle_plugin_panel',
    )
    if not p.hud.plugin_panel_open:
        return

    # Bundle header rows (trust tag) precede the plugin list so users can
    # see at a glance which bundles loaded, which were sandboxed, and
    # which had modules refused by the sandbox.
    for bundle in bundles:
        if sctx.y + _ROW_PANEL_H > p.H - _TAIL_GAP:
            break
        if bundle.load_errors:
            tag = 'refused'
            color = _COLOR_BUNDLE_REFUSED
        elif bundle.trusted:
            tag = 'trusted'
            color = _COLOR_BUNDLE_TRUSTED
        else:
            tag = 'sandboxed'
            color = _COLOR_BUNDLE_SANDBOXED
        label = _shorten(f'{bundle.name} [{tag}]', _NAME_MAX_CHARS + 8)
        sctx.text(label, sctx.col_x + 2,
                  sctx.y + theme.TEXT_BASELINE_ROW, color)
        sctx.y += _ROW_PANEL_H

    if not plugins:
        sctx.draw_text('no plugins found', color=theme.COLOR_HINT, indent=2)
        return

    max_rows = max(0, (p.H - sctx.y - _TAIL_GAP) // _ROW_PANEL_H)
    for plugin in plugins[:max_rows]:
        row = (sctx.col_x, sctx.y, sctx.col_w, _ROW_PANEL_H)
        sctx.add_hitbox(row, 'toggle_plugin', plugin.key)
        sctx.checkbox(sctx.col_x + _CHECKBOX_INSET_X,
                      sctx.y + _CHECKBOX_INSET_Y,
                      checked=plugin.enabled)
        color = (theme.COLOR_PLUGIN_ENABLED if plugin.enabled
                 else theme.COLOR_PLUGIN_DISABLED)
        sctx.text(_shorten(plugin.name, _NAME_MAX_CHARS),
                  sctx.col_x + _LABEL_INDENT,
                  sctx.y + theme.TEXT_BASELINE_ROW, color)
        sctx.y += _ROW_PANEL_H

    if len(plugins) > max_rows:
        sctx.draw_text(f'+{len(plugins) - max_rows} more',
                       color=theme.COLOR_HINT, indent=2,
                       height=_ROW_PANEL_H)


def register_sidebar(add):
    add('Plugins panel', _draw_plugins_panel, priority=400,
        key='builtin:plugins_panel')
