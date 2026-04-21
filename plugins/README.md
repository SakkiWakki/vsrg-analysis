# Plugins

This directory holds **bundles** — self-contained plugin packages that can
contribute replay overlays, sidebar sections, visualizations, and themes.
The shipped `builtin/` bundle is a reference implementation; anything you
drop alongside it follows the same layout.

> ⚠️ **The bundle layout and manifest format may change without notice
> until this note is removed.** Pin to a specific app version if you need
> stability.

## Layout

```
plugins/
  <your_bundle>/
    manifest.toml        # optional
    sidebar/             # HUD sections (register_sidebar(add))
      my_panel.py
    replay/              # lane-space draw plugins (register(add))
      my_overlay.py
    viz/                 # library-tab visualizations (register(add))
      my_plot.py
    theme/               # optional; overrides theme tokens when active
      __init__.py
```

A directory is recognized as a bundle if it contains at least one of
`sidebar/`, `replay/`, `viz/`, or a manifest alongside those folders.
Files whose names start with `_` are ignored (use them for shared helpers
— see `builtin/viz/_common.py`).

## Discovery

Bundles are picked up from, in order (later wins for duplicate keys):

1. The repo's `plugins/` directory — where `builtin/` lives.
2. `$EA_PLUGINS_PATH` — a colon-separated list of extra bundle roots.
3. `~/.config/vsrg-analysis/plugins/` — per-user bundles.

## `manifest.toml`

All fields are optional. Missing values fall back to the folder name.

```toml
name = "Sussy Baka"           # display name shown in menus
key = "sussy_baka"            # unique id; defaults to folder name, lowercased
version = "0.1.0"
author = "you@example.com"
```

## Role contracts

### `sidebar/*.py`
Each module exposes `register_sidebar(add)` and calls
`add(name, draw_fn, priority=…, pin_bottom=…)`. The draw function receives
a `SidebarContext` — see `analysis/player/sidebar_api.py` for the full
drawing vocabulary (`draw_button`, `draw_text`, `checkbox`, `split_row`,
…). Lower priority renders higher in the sidebar; pinned-bottom sections
hug `p.H`.

### `replay/*.py`
Each module exposes `register(add)` and calls
`add(name, draw_fn, stages=…, priority=…)` where `stages` is one or more
of `Stage.AFTER_LANES`, `AFTER_JUDGMENT`, `AFTER_NOTES`, `AFTER_GHOSTS`,
`HUD`, `POST_FRAME`. The draw function receives `(ctx, stage)` where
`ctx` is a `RenderContext`.

### `viz/*.py`
Each module exposes `register(add)` and calls
`add(name, builder, category='chart')`. `builder(replay, game, **kw)`
returns a matplotlib `Figure` or a `QWidget`. The category decides where
it shows up in the Visualize menu.

### `theme/`
If your bundle ships a theme, create `theme/__init__.py` (or `theme.py`)
with uppercase token names matching `analysis/player/theme.py`. Missing
tokens fall through to the built-in defaults. Only one theme is active at
a time.

## Minimal example

```
plugins/sussy_baka/
  manifest.toml
  sidebar/hello.py
```

`sidebar/hello.py`:

```python
def _draw(sctx):
    sctx.spacer()
    sctx.draw_heading('Sussy')
    sctx.draw_button('Click me', 'hello_click')


def register_sidebar(add):
    add('Hello', _draw, priority=500)
```
