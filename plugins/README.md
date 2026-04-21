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
a `SidebarContext`. Lower priority renders higher in the sidebar;
pinned-bottom sections hug `p.H`.

Two drawing styles are available:

- **Declarative (recommended for plugins):** build a `Component` tree
  from `analysis.ui` (`Column`, `Row`, `Heading`, `Text`, `Button`,
  `Checkbox`, `Spacer`, `Box`) and hand it to
  `analysis.ui.render_sidebar.render(sctx, tree)`. This is the only
  drawing surface reachable from sandboxed bundles.
- **Imperative:** call `SidebarContext` primitives directly
  (`draw_button`, `draw_text`, `checkbox`, `split_row`, …). This is how
  the built-in sections draw; it gives finer control but requires full
  Python access, so it's only available to trusted bundles
  (`builtin/`, `unsafe/`).

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

## In-game overlay (gamescope)

For games on Linux, we run through `gamescope`, a plugin
can publish a HUD that renders on top of the game window. The overlay
architecture is:

- A single C renderer (`analysis/games/osu/gamescope_overlay/osu_overlay`)
  attaches to a shared-memory widget array and draws rects + bitmap
  text. It knows nothing about the plugin's semantics.
- Plugins publish widgets from Python via
  [`plugins.overlay_api.OverlayPublisher`](overlay_api.py). One
  publisher = one shm segment = one feed the renderer can attach to.

Each publisher gets its own feed at `/dev/shm/vsrg_overlay_<plugin_key>`,
so multiple plugins can run side-by-side without trampling each other.
Launch the renderer with `--feed <path>` to point it at yours.

```python
from plugins.overlay_api import OverlayPublisher, WHITE, BLACK_DIM

pub = OverlayPublisher('my_plugin', width=2560, height=1440,
                      config_store=cfg)  # optional; persists drag layout
pub.start()

def build(pub):
    with pub.frame() as f:
        with f.group('panel'):              # drag as one piece
            f.rect('bg', 0.02, 0.02, 0.30, 0.18, color=BLACK_DIM)
            f.text('hello', 'HELLO', 0.03, 0.04,
                   px_scale=2.5, color=WHITE)

pub.run_thread(build, hz=30.0)
```

Key properties:

- **Normalized coords.** `(x, y, w, h)` are `[0, 1]` of canvas size;
  `anchor` (TL/TR/BL/BR/C) pins a corner so widgets reposition
  sensibly across resolutions.
- **Fixed-slot layout.** Up to 32 widgets per frame. Shrinking frames
  zero out trailing slots automatically.
- **Drag-to-reposition.** The renderer's Shift+Tab edit mode lets the
  user drag widgets. Widgets inside a `group(…)` block drag as one.
  If a `ConfigStore` is passed to the publisher, drag offsets persist
  per `(width × height)` bucket across restarts.
- **Ref publisher:** [`plugins/unsafe/osu_live/shm_publisher.py`](unsafe/osu_live/shm_publisher.py).

The overlay API lives at the bundle top level (`plugins.overlay_api`,
not under a role directory) because it's a shared runtime that
individual plugins drive — same idea as `analysis.plugins.host_api`.
It is **not** reachable from sandboxed bundles: publishing writes to
`/dev/shm` and needs `os`/`mmap`/`struct`, all blocked by the sandbox.
Use this API only from `plugins/builtin/` or `plugins/unsafe/`.

## Trust and sandboxing

Two trust levels:

| Location | Trust | Access |
|---|---|---|
| `plugins/builtin/` | trusted | full Python — ships with the app |
| `plugins/unsafe/<bundle>/` | trusted | full Python — opt-in escape hatch |
| Anywhere else (`plugins/<bundle>`, `$EA_PLUGINS_PATH`, `~/.config/…`) | sandboxed | restricted imports + stripped builtins |

### Sandboxed bundles

Sandboxed plugins run with a **restricted `__builtins__`** (no `open`,
`exec`, `eval`, `compile`, `input`, `breakpoint`, `memoryview`, frame-walk
primitives) and an **import allow-list**. Only these modules may be
imported:

- **Stdlib (pure):** `math`, `cmath`, `random`, `statistics`,
  `dataclasses`, `typing`, `enum`, `abc`, `collections`, `itertools`,
  `functools`, `operator`, `re`, `string`, `textwrap`, `bisect`, `heapq`,
  `array`, `copy`, `numbers`, `fractions`, `decimal`, `json`,
  `__future__`.
- **Third-party:** `numpy`.
- **Host API:** `analysis.player.theme`, `analysis.player.sidebar_api`,
  `analysis.player.plugin_api`, `analysis.player.events`,
  `analysis.plugins.host_api` (includes `plugin_config` — see below),
  `analysis.ui` (+ `analysis.ui.components`, `analysis.ui.render_sidebar`).

Anything else — notably `os`, `sys`, `pathlib`, `subprocess`, `socket`,
`urllib`, `requests`, `ctypes`, `threading`, `pickle`, `importlib` — is
refused. A refused module raises `SandboxViolation` at import time; the
bundle still loads, but the offending file is recorded in
`bundle.load_errors` and flagged in the Plugins sidebar panel.

**This is best-effort, not a security boundary.** NumPy in particular
has known escape vectors (`numpy.ctypeslib`). The goal is to stop lazy
harm and push plugin authors toward the host API — not to stop a
determined attacker. Only install bundles from sources you trust,
regardless of where they live in the layout above.

### The `unsafe/` escape hatch

If you're prototyping a plugin that needs raw Python (network calls,
filesystem access, threads, etc.), drop it under `plugins/unsafe/` to
skip the sandbox. The directory name is deliberate: naming it `unsafe`
makes the trust decision visible to anyone browsing the plugin layout.
Promote to `builtin/` once the plugin is ready, or request new host-API
surface if you're writing something for general distribution.

## Persistent per-plugin config

A plugin can persist its own settings through the shared config store.
All app config lives in one file — `~/.config/vsrg-analysis/config.json`
— under a nested tree:

```json
{
  "paths": {...},
  "plugins": {
    "mybundle:hello": {
      "replay_disabled": false,
      "settings": { "volume": 0.5, "colors": {"fg": "#fff"} }
    }
  }
}
```

Your plugin's settings live under `plugins.<your_key>.settings`. Reach
them through a scoped handle:

```python
from analysis.plugins.host_api import plugin_config

_cfg = plugin_config('mybundle:hello')
_cfg.set('volume', 0.7)        # persist a field
_cfg.get('volume', 0.5)        # read with default

def _on_change(field, old, new):
    print(f'{field} changed: {old!r} → {new!r}')

_cfg.subscribe(_on_change)     # fires when your settings change
```

Writes are debounced (bursts coalesce into one disk write) and fan out
to every running window — a config change made from one window's
dialog reaches the same plugin's instance in another window on the
next frame, with no restart.

The handle is scoped: one plugin can't reach another plugin's settings
or the top-level `paths.*` tree. This is a convenience boundary, not a
security boundary — trusted plugins could bypass it by touching the
store directly, but shouldn't.

## Minimal example

```
plugins/sussy_baka/
  manifest.toml
  sidebar/hello.py
```

`sidebar/hello.py` (declarative — works in sandboxed bundles):

```python
from analysis.ui import Button, Column, Heading, Spacer
from analysis.ui.render_sidebar import render


def _build():
    return Column((
        Spacer(),
        Heading('Sussy'),
        Button('Click me', 'hello_click'),
    ))


def _draw(sctx):
    render(sctx, _build())


def register_sidebar(add):
    add('Hello', _draw, priority=500)
```
