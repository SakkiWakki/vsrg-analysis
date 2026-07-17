# Engine-loop compiler restructure

Branch `notitg-engine-loop` (off `notitg`). Replaces the NotITG modfile
compiler's template-table harvesting with ONE headless engine simulation
loop, so the compile path works for every modfile template (classic gat,
Mirin, FMS_Cat "Cat" framework, anything else) instead of encoding gat's
framework internals, and so the module pile shrinks to something matching
what the job actually is.

Semantic authority: `ENGINE_ORACLE.md` (engine file:line citations).
Executable reference: the gat chart's own Lua/XML
(`/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/UKSRT8/5. gat/lua/`) --
the classic template's mod reader, UpdateCommand drivers, and proxy rigs
are implemented there; when a semantic question arises, read the chart's
implementation, don't guess.

## Why (the diagnosis)

The current compile is three imperative passes over one shared mutable
`StubEnvironment` (load -> `replay_mod_actions` -> `run_update_integration`),
each a later patch over the original bet of harvesting gat's
template-internal tables (`mods`/`mods2`/`mod_actions`). That boundary is
template-specific by construction, and every gap it caused got its own
mechanism: the replay pass, the frozen-recording/live_poke accumulator
dance, `aft_drivers.py` (a hand-transcribed gat closure, superseded by the
integrator but still shadowing it at `modfile._merge_driven_keyframes`),
`mod_channels`' hand-decoded clearall/revert semantics, `_SumTimeline` /
`_SpanGatedTimeline` gating, and `proxy_grid()`'s hardcoded `gat_*`
globals. `field_3d` even re-runs the whole pipeline privately a second
time. The engine API surface (`lua_api.VERB_REGISTRY`) was extracted last;
it should have been the foundation.

## Fixed decisions (user, 2026-07-17)

1. Output = the CURRENT compiled-dict contract. Downstream consumers
   (adapter surfaces, note_mods, field_instances, shader_bridge,
   storyboard) do not change in this arc. CompiledDocument migration
   continues separately per `render/DESIGN_compiled_document.md`.
2. Migration = parallel build, then cutover. The old path keeps working
   until the new one passes the gate (below); then the harvest machinery
   is deleted in one commit.
3. Mirin's declarative front-end (`mirin_compiler.py`) stays as a
   precision fast-path (exact analytic eases beat tick-sampled ones) and
   gets WIRED (it currently isn't -- the intended diff sits as a comment
   at `mirin_compiler.py:749-765`). Routing: mirin chart -> declarative
   front-end; everything else -> engine loop.
4. All work on branch `notitg-engine-loop`.

## Architecture

One new subpackage `analysis/games/notitg/sim/`:

- `actor.py` -- `SimActor`: the actor is simultaneously a SIMULATOR and a
  RECORDER. Reads (`GetX`, `getrotation`, ...) return the value at the
  current sim time INCLUDING in-flight tween interpolation and effect
  oscillator contribution (today's `RecordingActor.get` deliberately
  ignores both -- a whole fidelity class). Recording output stays compact:
  tween keyframes + analytic oscillator spans, exactly the shapes the
  element compiler already consumes. Grows a real per-actor command queue
  (SM runs queued commands next update; today's `advance_clock_by_pending`
  is an approximation -- expect justified diffs here). Verb dispatch stays
  driven by `lua_api` tables; `VERB_REGISTRY` remains the CI-firewalled
  contract (`test_lua_api.py` + `gat_called_methods.txt`).
- `env.py` -- `SimEnvironment`: engine singletons (GAMESTATE, SCREENMAN,
  MESSAGEMAN, ...), the Lua bridge (`__make_recorder` routing via
  generated `__GETTER`/`__COMMAND` sets -- carried over), actor registry,
  message/command dispatch with frame-boundary queues. No table
  harvesting, no recording freeze, no isolated-table detach.
- `loop.py` -- the main loop. Load pass (Init/On at the anchored load
  clock, ordering trap already solved), then tick t from load anchor to
  chart end at 60 Hz: set song beat/time clocks, drain queued
  commands/messages, run EVERY actor's UpdateCommand body verbatim.
  No `perframe` window regexing -- `perframe` is chart-defined Lua and
  gates itself. The chart template's own mod reader therefore runs for
  real: its per-tick `clearall` + `ApplyModifiers` calls ARE the mod
  stream (no `mods`-table normalization, no clearall decoding, no
  one-shot replay pass -- `mod_actions` fire when the template's own
  cursor reaches them, with recording live).
- `record.py` -- the recorded streams: per-actor keyframes/spans,
  the ApplyModifiers/ApplyGameCommand tick stream + coalescing (the
  existing `_coalesce_ticks` shape survives -- it is the right tool,
  158k -> 6k proven), shader flags, AFT/proxy binds, driven spans.
- `producers.py` -- recorded streams -> the compiled dict. Field copies
  generalize: ANY actor bound via `SetTarget`/`SetTexture(GetTexture())`
  becomes a copy whose transform composes its recorded parent chain
  (generic tree walk -- no `gat_allproxies`/`_PROXY_NAMES` hardcode;
  wrap/accumulator semantics come out right because the chart's own wrap
  code executed against true `GetX` reads). The compiled-visibility
  invariant is enforced HERE uniformly: every driven actor rests hidden
  outside its driven spans (one rule, not per-mechanism gates).

Approach-speed chase stays analytic and downstream: the loop records
`ApplyModifiers(target, speed)` calls; `render/mods/channels.py` keeps
compiling the chase. `render/mods/arrow_effects.py` untouched.

## Output contract (must keep emitting)

Live keys and their consumers (verified 2026-07-17): `mod_events`
(-> compile_mod_channels / compile_scroll_multipliers), `shader_flags`
(-> shader_bridge), `field_copies` + `base_field_hidden`
(-> NotitgFieldInstances / field_3d guard), `screen_transform`
(-> NotitgScreenCamera), `tree`/`elements` (-> storyboard),
`has_background` (-> background_path). Diagnostics: `unsupported`,
`named_actors`, `recorded_keyframes`, `replay`, `integration`,
`warnings`. Produced-but-unread today (emit if cheap, else drop with a
note): `aft_bg_visible`, `field_oscillators`, `field_vanish`,
`screen_oscillator`. `field_3d._player_field_keyframes` must be
re-pointed at the shared compile (kills the private second run).

## What dies at cutover / what survives

Deleted: `aft_drivers.py`; `update_integrator.py`; mod_stubs'
mods/mods2/mod_actions harvest + `replay_mod_actions` +
`run_update_integration` + freeze/live_poke; `modfile._normalize_*`,
proxy-grid producers, `_SumTimeline`/`_SpanGatedTimeline`;
`mod_channels`' clearall/window resolution (its modstring parser
survives as the ApplyModifiers parser).

Survives: `lua_api.py`, `xml_actors.py`, `sprite_sheet.py`, bitmap
fonts, LuaHost, modfile's generic layers (FGCHANGES parse, include
splice, asset/font resolution, element-tree compilation, oscillator
keyframe synthesis), `render/mods/*`, `field_instances.py`,
`shader_bridge.py`, `note_mods.py`, all adapter surfaces,
`mirin_compiler.py`.

With the harvest gone, `modfile.py` (1669 lines) splits along its
existing seams: `chart_doc.py` (FGCHANGES/lua-dir/XML/include/timing),
`assets.py` (texture/sprite/font resolution), `elements.py` (element
tree + oscillator synthesis), `compile.py` (thin orchestrator:
chart_doc -> sim.loop -> producers -> dict). `mod_stubs.py` dissolves
into `sim/env.py` + `sim/record.py`.

## Phases (each lands green on the branch)

1. `sim/actor.py`: value-at-t sampling (tween + oscillator), command
   queue, recording outputs. New semantics tests cited to ENGINE_ORACLE;
   port the RecordingActor tests that pin engine (not harvest) behavior.
2. `sim/env.py` + `sim/loop.py` + `sim/record.py`: gat runs end-to-end
   under the loop; report ticks/faults/coalesced-window counts.
3. `sim/producers.py` + a diff harness (`tools/compile_diff.py`): old vs
   new compiled dicts compared channel-by-channel at sampled t
   (positions ~1px, rotations ~0.5deg, alpha ~1% -- the ENGINE_ORACLE
   trace-diff tolerances), plus `lifetime_lint` on both. Differences are
   acceptable ONLY where the loop is more engine-faithful, each one
   written down.
4. Parity gate: new path behind an opt-in flag on the adapter;
   `tools/gat_oracle.py` at the 9 baseline timestamps
   (t = 8, 42, 150, 236, 253, 322, 346, 383, 415) structurally stable vs
   `refs/notitg/oracle_baseline`; plus one NON-gat classic-template chart
   from the local library as a generality smoke (pick during this phase).
5. Cutover: flag flips to default, harvest machinery deleted, harvest-
   pinned tests rewritten against the loop (`test_notitg_update_integrator`,
   `test_notitg_mod_channels`, the harvest block of `test_notitg_modfile`,
   `test_notitg_field_visibility`), modfile split lands.
6. Mirin wiring: `is_mirin_chart` routing in the orchestrator
   (declarative fast-path), reconciling the two output-dict shapes.

## Verification

Full suite (~1362) green at every phase; oracle montages structurally
identical at the 9 timestamps except justified improvements; compile-diff
report reviewed at phases 3-4; second-chart smoke at phase 4; compile
time within ~2x of today's ~7s (cached path unchanged).
