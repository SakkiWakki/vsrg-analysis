This is entirely vibecoded because I'm just making this as a side project to analyze my own gameplay and what I can change to accommodate my RSI. Though I made the clanker model some of its code based off previous work I had made. I also designed the architecture for the program and just told the clanker what to do. Specifically I told the clanker to allow user analysis plugins using python files and also to make implementation game-agnostic.

DO NOT USE THIS IN PRODUCTION SYSTEMS. I REPEAT, DO NOT USE THIS IN PRODUCTION SYSTEMS. THE CODE IS NOT VERY MODULAR AND PROBABLY HAS A TON OF BUGS FROM THE CLANKER VIBE CODING.

Use it for personal use if you want lol. I might rewrite parts by hand if really necessary, I just wanted something working rather than clean and correct.

Lemme know if the clanker somehow made a buffer overflow in python code lol

Btw I added plugins but the plugins themselves are only cheaply sandboxed. This means you should not run arbitrary unsafe plugins from randos

---

# Clanker README

A Python toolkit for offline analysis of **Etterna** and **osu!mania** replays. It reads replay files directly off disk, aligns them against the source chart, and produces timing statistics, plots, HTML reports, and a scrollable + playable replay viewer. Works with any keycount (4K, 7K, 9K, …).

## Features

- **Unified replay library** — auto-discovers Etterna profiles (`ReplaysV2` + `Etterna.xml`) and osu! installs (`Data/r/*.osr` + Songs dir), merging scores into one searchable list. First-run prompt lets you point it at custom install paths, and you can change them later from the Library tab's **Paths…** button.
- **Per-note timing analysis** — mean/std offset, judgments, hand splits, per-column drift, rolling stability, chord-size timing, coupling (solo vs paired notes).
- **Bundled plugin system** — visualizations, sidebar HUD sections, lane-space draw overlays, in-game overlays, and themes all ship as plugin **bundles** under `plugins/`. Sandboxed bundles get a restricted Python environment with an import allow-list; trusted bundles go under `plugins/builtin/` or `plugins/unsafe/`. See [plugins/README.md](plugins/README.md).
- **Embedded replay player** — native Qt/QPainter chart view, with audio sync, playbar scrubbing, scroll/rate controls, swappable note skins (bar/circle), draw-stage plugins, and **SV (scroll velocity)** support for osu!mania.
- **In-game overlay (Linux / gamescope)** — a standalone C renderer attaches to a shared-memory widget feed and draws a HUD on top of the running game. Plugins publish widgets through `analysis.overlay.api`; the host owns the `/dev/shm` publisher. Shipped adapter: the osu live bundle in `plugins/unsafe/osu_live/`.
- **HTML report export** — single-file self-contained summary report with all plots embedded as base64.
- **Batch mode** — run analysis across every score in a profile and produce leaderboards / cross-chart comparisons.

## Previews

Embedded replay player (osu!mania 10K):

![Player preview](player_preview.png)

Full-report export (osu!mania 10K — *xi - Aragami*):

![Example report](report.png)

## Requirements

- **Python 3.10+**
- [`numpy`](https://numpy.org/) — array math
- [`matplotlib`](https://matplotlib.org/) — plotting + report generation
- [`osrparse`](https://pypi.org/project/osrparse/) — osu! `.osr` parser
- [`PySide6`](https://pypi.org/project/PySide6/) — Qt GUI

Pitch-preserving rate changes use an in-house numpy phase vocoder — no
extra audio deps.

Install with pip:

```bash
pip install -r requirements.txt
```

### Setup from a fresh clone

The toolkit runs on Linux, macOS, and Windows. The Python-only path is the
same everywhere — the bundled Makefile also builds optional native pieces
(osu memory reader, gamescope overlay) on Linux.

```bash
git clone https://github.com/<you>/vsrg-analysis.git
cd vsrg-analysis

# --- Option A: Linux one-shot via Makefile -------------------------------
# Builds venv + installs deps + compiles the native osu memory reader + the
# gamescope overlay binary, then launches the GUI.
make

# --- Option B: Python-only (any OS) --------------------------------------
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (cmd)
# .venv\Scripts\Activate.ps1         # Windows (PowerShell)
pip install -r requirements.txt
python -m analysis.gui.app
```

Useful Makefile targets (Linux): `make venv` (just the venv),
`make native` (Rust PyO3 osu memory reader), `make overlay`
(C gamescope overlay binary), `make test`, `make clean`, `make distclean`.

On first launch you'll be prompted for your osu! Songs folder and Etterna
Save folder — both optional, both editable later via **Library → Paths…**.

**Platform notes:**

- **Linux** — the GUI uses Qt. Most desktop distros work with the PySide6
  wheels directly; on minimal installs, add your distro's Qt/XCB runtime
  packages if the app fails to create a window.
- **macOS** — everything installs via pip directly. On Apple Silicon make
  sure you're on Python 3.11+ so the arm64 PySide6 wheels are used.
- **Windows** — no extra system deps; all wheels ship with their DLLs. Use
  the `python` launcher or `py -3.11 -m venv .venv` to pin a specific
  Python version if you have multiple installed.
- **osu!-on-Wine (Linux)** — the autodetect looks in `~/.local/share/osu-wine/`
  and the standard Lutris/Bottles paths, but if yours is elsewhere just
  point Library → Paths… at it manually.

### Optional: convenience launchers

The repo ships with a `run-gui.sh` shell script for Linux/macOS and an
`analyze` CLI entrypoint. Both call `python3` — make sure your venv is
activated, or invoke them through the venv's Python directly
(`.venv/bin/python -m analysis.gui.app` / `.venv\Scripts\python -m analysis.gui.app`).

On Linux, `make` (or `make run`) will also launch the GUI after rebuilding
whatever's out of date.

## Running

```bash
# GUI (recommended — library browser, embedded player, all plugins)
./run-gui.sh                                         # convenience shell script
python -m analysis.gui.app                           # direct
make                                                 # Linux: build + launch

# CLI dispatcher (analysis without the GUI)
./analyze --help
./analyze replay /path/to/replay.osr                 # stats + plots for one replay
./analyze batch                                      # leaderboard across a profile

# Replay player standalone (Qt)
python -m analysis.player.player /path/to/replay.osr                 # osu!mania
python -m analysis.player.player /path/to/replay.bin --sm chart.sm   # Etterna
```

App state (install paths, per-plugin settings, window geometry, scroll
speed, etc.) is persisted in `~/.config/vsrg-analysis/config.json`. Writes
are debounced and changes fan out live to every open window — no restart
needed. Qt-specific UI state (geometry, filter/sort choices) still uses
`QSettings` alongside the JSON config.

### First run

On first launch you'll get a dialog asking for your **Etterna Save folder** and
**osu! Songs folder**. Both are optional:

- Leaving a field blank falls back to the usual autodetect paths (`~/.etterna/Save`,
  `~/osu!/Songs`, `~/.local/share/osu-wine/osu!/Songs`, etc.).
- Either field can be re-edited any time via **Library → Paths…**; changing it
  triggers a cache refresh so the library re-scans the new root.
- osu! replays are picked up from the `Data/r/` folder adjacent to the configured
  Songs dir, plus the default Wine/native locations.

## Testing

A pytest suite lives under [tests/](tests/). Tests run against an isolated
`XDG_CONFIG_HOME` so they don't touch your real `QSettings`.

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

Currently covered: install-path overrides, validators, `find_*_dirs` override
precedence, `PathsDialog` save/clear/prefill behavior, and first-run prompt gating.

## Project layout

```
analysis/                         (Python package)
├── games/                        per-game code behind a uniform adapter
│   ├── etterna/
│   │   ├── adapter.py            resolves audio + chart, judgment windows
│   │   ├── replay.py             .bin parser + Etterna.xml score rows
│   │   └── sm_chart.py           .sm / .ssc parser
│   └── osu/
│       ├── adapter.py            .osu alignment + SV integration + audio lookup
│       ├── replay.py             .osr parser + songs-dir discovery
│       ├── native/               optional Rust PyO3 osu memory reader
│       └── gamescope_overlay/    C external-overlay renderer (shm attach)
├── core/                         cross-game logic
│   ├── game.py                   adapter registry + common replay dict shape
│   ├── search.py                 unified library scan across both games
│   ├── timing.py                 per-column/hand stats, chord detection, drift
│   └── batch.py                  leaderboard-style analysis across a profile
├── viz/                          plotting
│   ├── plots.py                  matplotlib plotters + HTML report export
│   ├── note_visualizer.py        scrollable chart renderer (Note Viewer plugin)
│   └── plugins/                  legacy viz registry (delegates to analysis.plugins)
├── player/
│   ├── player.py                 replay player state/model + standalone launcher
│   ├── qt_renderer.py            native Qt/QPainter pipeline + plugin hook dispatch
│   ├── render_context.py         per-frame context passed to player draw plugins
│   ├── audio.py                  streaming phase vocoder (pitch-correct rate)
│   ├── culling.py                visible-window selection for notes/holds
│   ├── scroll.py                 scroll/SV math shared across adapters
│   ├── hud_state.py, events.py   HUD data feed + input events
│   ├── plugin_api.py             Stage enum for lane-space draw plugins
│   ├── sidebar_api.py            sidebar-section API + SidebarContext helpers
│   ├── theme.py                  UI design tokens (proxies the active theme)
│   └── plugin_loader.py          plugin discovery + persistence + dispatch
├── overlay/                      in-game HUD runtime
│   ├── api.py                    sandbox-safe overlay helpers (colors, anchors)
│   └── publisher.py              /dev/shm feed publisher + edit-mode bridge
├── plugins/                      bundle discovery + sandbox
│   ├── host_api.py               plugin_config, shared registries
│   └── sandbox.py                import allow-list + restricted builtins
├── ui/                           declarative sidebar components
│   ├── components.py             Column/Row/Heading/Button/Checkbox/Spacer/Box
│   └── render_sidebar.py         render(sctx, tree)
├── config/                       shared JSON-backed config store
│   ├── store.py                  ~/.config/vsrg-analysis/config.json
│   └── migrate.py                legacy QSettings migration
└── gui/
    ├── app.py                    PySide6 main app — library, tabs, player, plots
    ├── settings.py               QSettings wrapper + install-path overrides
    ├── paths_dialog.py           first-run / edit-anytime install-path prompt
    ├── plugins_dialog.py         per-bundle plugin enable/disable UI
    ├── library_tab.py            score tree, filters, open-viz/player flows
    └── player_tab.py             embedded replay player tab

plugins/                          bundles (see plugins/README.md)
├── builtin/                      trusted; ships with the app
│   ├── viz/                      scatter, distribution, drift, heatmap, report, …
│   ├── sidebar/                  judgments, options, scroll, hints, status, plugins
│   └── replay/                   lane-space draw overlays
├── unsafe/osu_live/              trusted live-overlay adapter for osu!
└── example_sandboxed/            minimal sandboxed example

tests/                            pytest suite
analyze                           CLI entry point
run-gui.sh                        GUI launcher script
Makefile                          Linux build orchestration (venv + native + overlay)
```

## Plugin bundles

Plugins now ship as **bundles** — self-contained directories under
`plugins/` that can contribute visualizations, sidebar sections, lane-space
draw overlays, in-game overlay feeds, and themes. Minimum viz plugin:

```python
# plugins/my_bundle/viz/my_plot.py
from plugins.builtin.viz._common import clean_arrays, new_fig

def build(replay, game='etterna', on_play=None, **_):
    rows, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(10, 5)
    ax.plot(rows, offs * 1000)
    return fig  # or a QWidget for interactive plugins

def register(add):
    add('My custom plot', build, category='chart')
```

A directory is recognized as a bundle if it contains at least one of
`viz/`, `sidebar/`, `replay/`, or `overlay/` (plus an optional
`manifest.toml`). Bundles are discovered from `plugins/` in the repo,
`$EA_PLUGINS_PATH` (colon-separated extra roots), and
`~/.config/vsrg-analysis/plugins/`.

Trust model:

- `plugins/builtin/` — trusted, ships with the app.
- `plugins/unsafe/<bundle>/` — trusted opt-in escape hatch for things that
  need raw Python (network, subprocess, threads, `/dev/shm`).
- Anywhere else — **sandboxed**: restricted `__builtins__` plus an import
  allow-list (stdlib pure modules, `numpy`, and the `analysis.*` host APIs).

Built-in visualizations: timing scatter, offset distribution, per-column
means, per-hand drift, coupling, rolling stability, chord sizes,
column×offset heatmap, note viewer, full report (3×3 grid with plot picker).

See [plugins/README.md](plugins/README.md) for the full manifest, role
contracts (sidebar, replay, viz, overlay, theme), sandbox rules, and the
per-plugin persistent config API (`plugin_config(key)`).

## Replay data format

Both parsers emit the same dict shape so everything downstream is game-agnostic:

```python
{
    'noterows':  np.int64[n],    # osu: ms; Etterna: noterows (48/beat)
    'offsets':   np.float64[n],  # seconds, signed (negative = early)
    'columns':   np.int32[n],
    'notetypes': np.int32[n],
    'misses':    np.bool_[n],
    'holds':     [(start_row, col, end_row), …],
    'keycount':  int,
    'filepath':  str,
    # osu only:
    'chart_path':   str,
    'sv_sections':  [(time_sec, sv_multiplier), …],  # empty if chart has no SV
}
```

## Replay player

Keys (when the chart view has focus):

| Key | Action |
| --- | --- |
| Space / P | pause / resume |
| Left / Right | seek ±2s (Shift for ±10s) |
| Up / Down | scroll speed |
| + / - | playback rate |
| R | restart |
| mouse wheel | seek ±0.5s (Shift for ±5s) |

Bottom controls: play/pause, playbar (click anywhere to jump), scroll ±, rate ±, restart, and **SV toggle** (osu!mania only — disabled when the chart has no SV).

## Player draw plugins

Lane-space draw plugins live under a bundle's `replay/` directory and
expose `register(add)`:

```python
# plugins/my_bundle/replay/receptor_flash.py
from analysis.player.plugin_api import Stage


def draw(ctx, stage):
    # ctx exposes ctx.painter for native Qt drawing, plus t_now, keycount,
    # lane geometry, candidates, visible_ghost_holds, and helpers such as
    # time_to_y()/lane_center().
    if stage != Stage.AFTER_JUDGMENT:
        return
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QPen

    ctx.painter.setPen(QPen(QColor(255, 255, 255), 1))
    for col in range(ctx.keycount):
        ctx.painter.drawEllipse(
            QPointF(int(ctx.lane_center(col)), int(ctx.judge_y)), 5, 5)


def register(add):
    add('Receptor flash', draw, stages=[Stage.AFTER_JUDGMENT], priority=100)
```

Available stages: `AFTER_LANES`, `AFTER_JUDGMENT`, `AFTER_NOTES`,
`AFTER_GHOSTS`, `HUD`, `POST_FRAME`. Lower priority runs earlier within the
same stage. Plugins outside `plugins/builtin/` and `plugins/unsafe/` run
sandboxed — if you need `PySide6`, put the bundle under `unsafe/`.

The replay player's right sidebar has a collapsible `Plugins` panel — open
it to enable or disable any discovered plugin. Choices persist in the
shared config store under `plugins.<bundle>:<key>.replay_disabled`.

## Known limitations / caveats

- Etterna `.bin` offsets are quantized relative to judgment; parsing replicates the game's `GetOffset` interpretation but may disagree with in-game grade display at the edges.
- osu!mania note→press alignment is greedy-nearest within a ±188ms window. Heavily ghost-tapping scores may misattribute presses.
- At non-1x rates, pitch correction is handled by a small numpy phase vocoder (RTPGHI-style heap integration). Output is ~1 dB quieter on dense broadband material at rate ≠ 1 because reconstructed phases OLA less coherently than the analysis phases — audible as a slight loudness dip, not a bug in the math. Turning pitch correction off falls back to simple resampling, so pitch shifts with rate.
- No Windows-style file lock handling for `Etterna.xml` — close the game before scanning.
- This is a personal side project. See the warning above.
