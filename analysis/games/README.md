# Games

Each subdirectory here is a game the analyzer knows how to load replays
for. At startup, `analysis.core.game.discover_games()` scans this directory,
imports every `<game>/adapter.py`, and registers the exposed adapter.

## Adding a game

1. Create `analysis/games/<game>/` with an empty `__init__.py`.
2. Add `replay.py` (or whatever you need) implementing a parser that returns
   the usual dict with `noterows`, `columns`, `offsets`, `misses`,
   `notetypes`, `holds`, and optionally `keycount`, `chart_path`, `od`,
   `sv_sections`, `ghost_taps`, `miss_holds`, etc.
3. Add `adapter.py` exposing an `ADAPTER` instance of a `GameAdapter`
   subclass (see `analysis/core/game.py`):

   ```python
   from analysis.core.game import GameAdapter
   from analysis.player import scroll

   class MyGameAdapter(GameAdapter):
       name = 'mygame'

       def parse_replay(self, path, chart_path=None): ...
       def resolve_audio(self, replay, entry=None, progress=None): ...
       def judgement_windows(self, replay, **_): ...
       def judge_label(self, replay, **_): ...
       def default_scroll_mode(self): return 'mygame-scroll'
       def player_kwargs(self, replay, **_): return {}

   # Scroll modes are global — registering them here makes them available
   # to players regardless of which game the current replay is from, so
   # users can compare speeds across games.
   scroll.register(scroll.ScrollMode(
       key='mygame-scroll',
       label='MyGame SV',
       game='mygame',
       to_pxps=lambda value, opts, p: ...,   # px/sec at player.H
       from_pxps=lambda pxps, opts, p: ...,  # inverse
       default_value=10.0,
       value_bounds=(1.0, 100.0),
       nudge=scroll.multiplicative_nudge,    # or integer_step_nudge, ms_nudge
       format_value=lambda v: f'MG {v:.1f}',
       options={'mini': 0.0},                # per-mode knobs, optional
       on_enter=None, on_exit=None,          # lifecycle hooks, optional
   ))

   ADAPTER = MyGameAdapter()
   ```

4. That's it — no registry files to edit. The game's modes appear in the
   player HUD's scroll-type cycle next to the others.

## Conventions

- Scroll modes should express their formulas against a 480-tall logical
  playfield (`Player.REFERENCE_FIELD_H`). The Player scales by
  `H / REFERENCE_FIELD_H` so cross-game comparisons work (e.g. Etterna
  C952 ≈ osu SS 30, matching cmodcalc.com).
- Store per-mode knobs (mini, receptor size, custom SVs) in the mode's
  `options` dict with sensible defaults. Users access them via
  `player.get_mode_option` / `player.set_mode_option`.
- Use lifecycle hooks (`on_enter` / `on_exit`) for side effects that need
  to survive the mode switch — CMOD uses them to suspend SV on entry and
  restore it on exit.
