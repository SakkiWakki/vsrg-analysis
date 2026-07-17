# Mirin Template alignment

How the Mirin Template (XeroOl/mirin-template, `template/template.lua`
~1667 lines + `template/ease.lua`) maps onto our compiled-document /
mod-channel model, what it confirms vs corrects in our
reverse-engineered classic-template semantics, and how to add a Mirin
front-end.

Source studied (cloned to scratchpad): `template/template.lua` (the
applier), `template/ease.lua` (easing library), `template/std.lua`
(`perframe_data_structure`, environments), `spec/*_spec.lua` (executable
semantics). Docs: xerool.github.io/mirin-template.

Our side: `analysis/games/notitg/mod_channels.py` +
`analysis/player/render/mods/channels.py` (approach-chase compiler),
`note_mods.py`, `modfile.py`/`mod_stubs.py` (classic-template recorder),
`render/DESIGN_compiled_document.md` (target ontology).

KEY UP-FRONT FACTS:
- Mirin is Lua we can run once under our LuaHost. Its runtime data
  structures (`eases`, `funcs`, `nodes`, `auxes`, `default_mods`) are
  plain tables, harvestable after `ready_command`.
- Our existing NotITG pipeline reverse-engineered the CLASSIC template
  (gat's `default.xml`: per-frame `clearall` + `ApplyModifiers` string
  reader), NOT Mirin. Mirin's applier is a DIFFERENT, cleaner model that
  CONFIRMS most of our approach-chase math and CORRECTS one thing.
- The declared end-goal pilot "The Government Knows" is NEITHER classic
  NOR Mirin - it is a bespoke `@include`-built "Cat" framework
  (`fg/build.js`, `CatUpdater`/`CatCommand`/`CatShader`/`CatAFT`,
  `GAMESTATE:LaunchAttack` mod strings). See section 2.

---

## 1. Semantics map: Mirin concept -> our system

### 1.1 ease / add / set / acc -> ModEvent rows

Mirin call shapes (`template.lua` L146-219, verified in `ease_spec.lua`):

    ease {beat, len, easefn, pct, 'mod', [pct2, 'mod2', ...], plr=, m=/mode=, time=}
    add  {...}                 -- self.relative = true; else identical to ease
    set  {beat, pct, 'mod'}    -- inserts (len=0, ease=instant) -> ease
    acc  {beat, pct, 'mod'}    -- relative set: (len=0, instant, relative)

Mapping to our `ModEvent(beat, value, speed, mod, player)`:
- `beat` -> event time. Mirin stores `start_time` in SECONDS at declare
  time via `song:GetElapsedTimeFromBeat(self[1])` (L179) unless `time=`.
  So Mirin's beat->time conversion is the SAME job our
  `ModChannels.compile(beat_to_time=...)` does; harvest `start_time`
  directly (already seconds) and use identity clock, exactly like
  `mod_channels.py` does for gat windows.
- `pct` -> `value/100` (Mirin percent is 0..100+, ours is a fraction).
- `easefn` -> our `_Segments` are piecewise-LINEAR. Mirin eases are
  arbitrary curves (see 1.7). This is the ONE real IR gap: our channel
  compiler cannot represent `outExpo` faithfully - it would linearize.
  Two options in the front-end: (a) bake each ease into densely-sampled
  breakpoints, or (b) extend `_Segments` to carry an ease-id per segment
  (osu enum) and evaluate the curve at sample time. See 3 and 4.
- `speed` (`*S` approach) has NO Mirin analog. Mirin has no approach
  chase at all (see 1.8 CORRECTION). A harvested Mirin ease is a pure
  keyframe pair `(start_time, from) -> (start_time+len, to)`. Set
  `speed` unused; emit two breakpoints directly.

RELATIVE vs ABSOLUTE (the crux, `run_eases` L890-1017):
- Mirin keeps a `targets[pn][mod]` table = "value the mod will hold when
  all active eases finish". `ease` (absolute) at activation time
  computes its relative delta `e[i] = e[i] - targets[plr][mod]` (L966)
  so it animates FROM the current committed target TO the absolute pct.
  `add`/`acc` (relative) skip that subtraction (`if not e.relative`),
  layering a delta on top. `set` = instant absolute; `acc` = instant
  relative.
- All active eases SUM into the live value every frame:
  `mods[plr][mod] = mods[plr][mod] + e3 * e[i]` (L992). Multiple eases
  on one mod stack additively while animating, then commit to `targets`.
- "Transient vs sticky": `e[3](1) >= 0.5` decides whether an ease
  permanently moves the target (`offset=1`, L948-949) or is a transient
  bump that returns to the prior target (e.g. `bounce`, `spike`, `pulse`
  from `ease.lua` all end at 0). This is the same "does the ease stick"
  question our `channels.py` handles via whether a later event restores
  rest, but Mirin decides it from the EASE FUNCTION's endpoint, not from
  a following window. The front-end must replicate: sample `easefn(1)`;
  if `< 0.5`, the ease returns to the pre-ease target (emit a keyframe
  back to `from`); if `>= 0.5`, it commits `to` as the new target.

Our per-target summation model already matches Mirin's additive stack
because `ModChannels` groups by `(mod, player)` and sums are implicit in
how gat re-applies. But Mirin's target/live split is cleaner than gat's
clearall-reapply; harvesting eliminates the need to model it - we read
the resolved timeline, not the mechanism.

### 1.2 plr targeting -> per-player channels

`plr` (L184-194): table -> one duplicated ease row per player number;
number -> that row's `plr`. Default `{1,2}` (L93). `plr` can be a global
in the xero env (`get_plr` L97). This is EXACTLY our
`ModEvent.player` split and `ModChannels.values_at(t, player)`. Mirin
players are 1..8 (`max_pn`); our convention is 0-based (`pn-1`), so the
front-end subtracts 1. Item 43's dual-player work consumes this
directly - Mirin charts are the canonical multi-player case
(`lua/mods.lua` hard-requires P1+P2).

### 1.3 aux + node -> computed channels / dataflow graph

- `aux {'mod'}` (L422): marks a mod so `run_mods` (L1126) does NOT emit
  it into the `ApplyModifiers` buffer. It is a template-internal
  variable. Maps to our "channel that has no engine formula" - we
  already keep unknown-named channels (mod_channels harvests them; the
  per-note pipeline ignores names it has no kernel for). An aux mod is
  just a channel consumed by a node, not by the engine.
- `node {inputs..., fn, outputs...}` (L436): a dataflow node run WHENEVER
  an input mod changes (`run_nodes` L1085 propagates from
  `node_start[k]` for every touched mod `k`). `fn(inputs, pn) ->
  outputs`. If a node reads and writes the same mod it OVERWRITES
  (terminator handling). The graph is COMPILED (L741-865) into a
  topologically-ordered set of generated closures with parent-output
  wiring - a real per-frame dataflow DAG.
- The zoom node (L1225) and `movex/xmod` definemods (L1256-1286) are
  built-in nodes: `zoom` fans out to `zoomx/zoomy`; `movex` fans out to
  `movex0..7` (per-column); `xmod` writes the speed modstring directly.

Compilability verdict:
- Nodes whose `fn` is a pure numeric function of input channels
  (`zoom`, `movex` repeat8, `blacksphereoffset` example -> invert/
  alternate/reverse) are DECLARATIVELY COMPILABLE: run the node fn over
  the sampled input curves on a time grid -> output channels. This is
  item 52's "chart-DEFINED mods compile to grid-sampled surfaces" and
  DESIGN_compiled_document axis 6 (computed streams). No integrator
  needed - they are pure functions of already-compiled channels.
- Nodes whose `fn` POKES AN ACTOR (`node {'rotatebg', function(p)
  my_bg_actor:rotationz(p) end}` from the docs) are actor bindings, not
  value transforms - they belong to our storyboard actor timeline tier
  (RecordingActor), driven by sampling the input channel. These are the
  bridge between mod channels and the actor tree; run them under the
  integrator sampling the input curve, recording pokes as keyframes -
  same machinery as `update_integrator.py`.

### 1.4 definemod -> chart-defined mod surface (item 52)

`definemod {'name', fn, 'out1', 'out2', ...}` (L473) = `aux('name')` +
`node`. Answering the task's explicit questions:
- DOES definemod get the ease value as input? YES. The eased percent of
  the aux'd mod IS the node input. When you `ease {0,1,linear,100,
  'blacksphereoffset'}`, the channel `blacksphereoffset` animates 0->100,
  and every frame the node fn receives that live percent and returns
  values written to `invert/alternate/reverse`.
- HOW sampled? Per FRAME, propagation-driven: only when an input
  changed (`run_nodes` walks `node_start[mod]` for touched mods). For
  our AOT compile: sample the input channel(s) on a time grid, run fn,
  emit output channels. Density = item 52's per-mod sampling hint.
- Shorthand form `definemod {'name', 100, 'a', 100, 'b'}` (node L437,
  the numeric branch) compiles to `function(p) return p*1.0, p*1.0 end`
  - a pure linear fan-out, trivially compilable to scaled copies of the
  input channel.

definemod is the DIRECT ANSWER to item 52 "Government Knows Lua compiles
to grid-sampled surfaces" and "covers mods that don't exist yet by
construction". Mirin formalizes it; we independently designed the same
shape. A Mirin `definemod` whose outputs are engine mods (invert, drunk)
compiles into extra `ModEvent`s on those channels; whose outputs are
further aux mods recurse; whose fn pokes actors -> storyboard tier.

### 1.5 func / perframe / func_ease / poptions -> update integrator

- `func {beat, fn}` (L263): one-shot scheduled Lua at a beat. = our
  scheduled `mod_actions` one-shots (item 16: gat's 667 closures fired
  once at compile). Run it once at its `start_time` under the host.
- `perframe {beat, len, fn}` (L321): fn called EVERY FRAME over
  `[beat, beat+len]`, receiving `(songbeat, poptions)`. This is the
  IRREDUCIBLE dynamic tail = our `update_integrator.py` (60Hz song-time
  ticks). A perframe that writes `poptions[pn].drunk = f(beat)` is a
  live channel source; sample it on the tick grid -> channel keyframes.
- `poptions[pn].mod = v` (L516-534): direct mod write bypassing the
  `*-1 v mod` string. `__newindex` writes `mods[pn][mod]` and logs the
  touch. So perframe's mod writes land in the SAME `mods` table eases
  write - unified. For us: a poptions write during integrator sampling
  is a channel keyframe, identical to how `mod_stubs` records
  ApplyModifiers pokes today.
- `func_ease {beat, len, ease, [from,] to, fn}` (L354): sugar = a
  perframe that feeds `from + (to-from)*ease(progress)` to `fn` each
  frame, plus a persist tail. Compiles like perframe.

Verdict: eases/add/set/acc/definemod-of-pure-fn/node-of-pure-fn are ALL
declaratively compilable from harvested tables (no ticking). Only
`func`/`perframe`/`func_ease` and actor-poking nodes need the
integrator - and we already have it (`update_integrator.py`,
`run_update_integration`).

### 1.6 get() readbacks

Not a distinct template export in this version; readback happens via the
`poptions[pn].mod` __index (L522, returns live `mods[pn][mod]`) and via
`xero.P[pn]` actor handles. A perframe reads current mod values through
poptions and reacts. For AOT this is fine: during integrator sampling
the live `mods` table is populated by the eases we already ran, so a
perframe reading `poptions[pn].drunk` sees the correct sampled value.
Deterministic per (chart) - no live input.

### 1.7 Easing library -> our Keyframe easing ids (osu enum)

`ease.lua` defines the full library. Enumerated (name -> whether we have
an equivalent in the osu Keyframe enum our RecordingActor maps to):

Standard in/out/inOut/outIn x quad/cubic/quart/quint/expo/circ/sine/
bounce/back/elastic -> the osu enum HAS in/out/inout for most of these
(Quad Cubic Quart Quint Sine Expo Circ Back Bounce Elastic). Coverage is
good for the "power" and named families.

Mirin-SPECIFIC eases WE LACK (no osu-enum equivalent; must port the Lua
verbatim or bake to samples):
- `instant` (L168, returns 1) - trivial, = a step. We have this concept.
- `linear` - have it.
- `bounce(t)=4t(1-t)`, `tri`, `bell`, `pop`, `tap`, `pulse`, `spike`,
  `inverse` - TRANSIENT bumps (end at 0, i.e. `f(1)<0.5`). NONE exist in
  osu enum. These are the ones that make Mirin charts "pulse" a mod.
- `popElastic`, `tapElastic`, `pulseElastic`, `impulse` -
  parameterized (with1param/with2params, L77/104) - LACK.
- Elastic/Back are PARAMETERIZED in Mirin (`inBack` default a=1.70158,
  `outElastic` a=1,p=0.3) and support `.param(x)` overrides; osu enum
  Back/Elastic are fixed-parameter. Default-param cases match; custom
  params LACK.
- `flip(fn)` (ease.lua L23) = `1-fn(x)`; `blendease(f1,f2)` (L35) =
  smoothstep mix. COMBINATORS producing new eases - not enumerable.

CONCLUSION: our osu-enum keyframe easing is INSUFFICIENT for Mirin. The
robust path is to evaluate the ACTUAL Lua ease function (we run the
template anyway) and bake to dense breakpoints, OR carry a callable per
segment. Do NOT try to map Mirin eases onto osu ids - the transient
family and combinators have no ids and are the whole point.

### 1.8 What Mirin CONFIRMS / CORRECTS in our reverse-engineering

CONFIRMS:
- beat->time conversion up front (`song:GetElapsedTimeFromBeat`) =
  our `beat_to_time` clock. Correct.
- Per-player rows, additive mod summation across active
  eases/perframes -> one `mods[pn]` buffer -> one `ApplyModifiers`
  string per player per frame (`run_mods` L1118-1139). Our
  per-(mod,player) channel model + summation matches.
- One-shot funcs fired at their beat; perframe as the dynamic tail =
  our compile-time one-shot replay + integrator split (items 16/59).
- aux mods (kept but not sent to engine) = our "unknown channel kept,
  ignored by kernels" behavior.

CORRECTS (the highest-value finding, see report):
- **Mirin has NO approach-speed chase and NO clearall.** It calls
  `ApplyModifiers('clearall')` EXACTLY ONCE at init (L1221) and
  thereafter emits `*-1 <pct> <mod>` (L1127) - the `*-1` means
  SNAP/instant approach every frame, because Mirin does its OWN easing
  in Lua and hands the engine the already-eased value. Our entire
  `channels.py` approach-chase compiler (fapproach, clearall-resets-
  speed-to-1.0, float-back-at-speed-1.0) models the CLASSIC gat reader's
  behavior and is correct FOR THAT, but is IRRELEVANT to Mirin. A Mirin
  front-end must NOT run the approach-chase compiler; it emits keyframes
  from the eased curve directly. Our item-31/158 "receptor snap-back"
  bug is a classic-template revert-dynamics artifact that DOES NOT EXIST
  in Mirin charts (Mirin never reverts via engine approach; the ease
  function's own return-to-0 does it). This bounds that bug to classic
  charts and tells us the two front-ends need DIFFERENT channel
  builders.
- Mod SUMMATION: our model sums per-frame contributions; Mirin confirms
  additive stacking is correct (L992) AND clarifies the target/live
  split (committed target vs animating overlay) that gat's clearall-
  reapply obscured. We had it functionally right by a different route.

---

## 2. Compilation strategy

### 2.1 Direct declarative harvest (the "better way")

The template IS Lua we execute once under LuaHost(luajit21). After the
chart's `mods.lua` / Module files run inside `ready_command`
(L1173-1199), these tables hold the compiled program:
- `eases` (L38): sorted timeline of every ease/add/set/acc/reset row,
  normalized (`sort_tables`/`resolve_aliases` L657/691 already ran).
- `funcs` (L44): scheduled funcs + perframes.
- `nodes` (L58): compiled dataflow graph (post `compile_nodes`).
- `auxes`, `default_mods`, `aliases` - the rest of the program state.

Reading these POST-INIT is sufficient for the declarative subset
(ease/add/set/acc + pure nodes + definemod fan-outs). We do NOT need to
stub `ApplyModifiers` for those - the eased timeline is fully described
by `eases` + the ease functions. This is cleaner than the classic
Mode-2 recorder (which stubs ApplyModifiers and watches the per-frame
string stream) because Mirin's declarative tables ARE the compiled form.

Two harvest levels:
1. HARVEST TABLES (preferred): stub `song:GetElapsedTimeFromBeat`,
   `GAMESTATE`, `SCREENMAN`, actor handles enough that `ready_command`
   completes; then read `eases`/`funcs`/`nodes`. For each ease, compute
   `(start_time, start_time+len)` and evaluate `easefn` -> breakpoints.
   Pure nodes/definemods run over sampled input channels.
2. TICK THE APPLIER (fallback, for func/perframe): additionally stub
   `ApplyModifiers`/`poptions` and drive `update_command` (L1202) on a
   60Hz song-time grid, intercepting the `mods[pn]` writes = channel
   samples. Reuse `update_integrator.py` verbatim. Only needed when
   `funcs` contains perframes with non-trivial (non-func_ease) bodies.

Because template.lua's exports live in the `xero` table and it uses
`setfenv`/`loadstring` heavily (node/func codegen L281,448,853), the
LuaHost sandbox must permit `loadstring` for the Mirin dialect (our
sandbox currently disables `load`/`require` - Mirin needs a controlled
`loadstring` + `setfenv`, luajit21 reserved slot already anticipated in
item 5). This is the main host change.

Serializable output: the harvested eases+nodes compile straight into the
CompiledDocument channel table (DESIGN_compiled_document axis 5/6) keyed
by chart hash - Lua runs once ever, cached.

### 2.2 Classic-template integrator path (contrast)

gat / UKSRT charts use the classic `default.xml` reader: no declarative
tables exist; the program is XML-CODE actors that build `mods`/`mods2`/
`mod_actions` data tables (item 8/16) read by a per-frame `clearall +
ApplyModifiers` loop. We already handle this via `mod_stubs.py` +
`modfile.py` + `update_integrator.py` + the approach-chase
`channels.py`. That path STAYS - it is a different (imperative,
approach-driven) template. Detection (2.4) routes to it.

### 2.3 What "The Government Knows" actually uses (task Q)

INSPECTED `/mnt/Yucky/.../FMS_Cat/The Government Knows [FMS_Cat]/`:
- `#FGCHANGES:0.000=fg=...` -> `fg/` dir, entry `fg/default.xml` (a
  BUILT/bundled file, 20KB) produced by `fg/build.js` from `fg/src/`
  (`@include(libs/...)`, `@include(mods/...)`).
- It is a BESPOKE "Cat" framework, NEITHER classic NOR Mirin: globals
  `CatEvent`, `CatUpdater`, `CatCommand`, `CatText`, `CatAFT`, `CatRNG`,
  `CatParticle`, `proxyguy`, `modguy`, `splineguy`, `cubicBezier`. Mods
  applied via `GAMESTATE:LaunchAttack(start, len, 'overhead, cel, 3x,
  zbuffer, 100 spiralholds, ...', pn)` (attack strings, a THIRD
  mod-application API distinct from ApplyModifiers) plus per-frame
  `CatUpdater.add(fn)` / `CatCommand.add(beat, fn)` callbacks
  (`fg/src/main.lua` is almost entirely per-frame closures - VHS frame
  logic, shader param writes).
- So it PREDATES/sidesteps Mirin and canNOT use a Mirin front-end. Our
  memory (item 28) already flags it as the 3D + per-actor-shader Stage-B
  pilot; the current recorder's "223 warnings / 7 elements" reflects the
  classic recorder choking on Cat's imperative per-frame framework -
  EXPECTED, since it is neither of our two known dialects. Government
  Knows needs EITHER a Cat-specific recorder OR (more likely) the
  generic integrator running its CatUpdater/CatCommand tick loop with
  LaunchAttack + Cat* stubbed. It is NOT a Mirin adoption target; treat
  it as its own third front-end later.

### 2.4 Detection: Mirin vs classic vs Cat

Cheap textual sniff on the FGCHANGES-referenced files before choosing a
compiler:
- MIRIN: presence of `setfenv(1, xero.strict)` / the `xero` table / the
  `ease {`/`add {`/`definemod`/`aux` export idiom, or a bundled
  `template.lua` / `mirin.lua` plugin. Strongest signal: the file
  `require`s or includes the Mirin template package.
- CLASSIC: `GAMESTATE:ApplyModifiers` string reader + `mod_insert`/
  `mod_message`/`clearall` idiom (gat) with no xero table.
- CAT / bespoke: `CatUpdater`/`LaunchAttack`/custom framework globals,
  `@include` build artifacts. Route to integrator-only (or defer).
Fallback = classic recorder (current default), which degrades to
warnings on unknown dialects (as Government Knows shows).

---

## 3. What we got right / wrong (ranked vs Mirin's documented model)

RIGHT (keep):
1. Per-(mod,player) channel decomposition + additive summation +
   `values_at(t, player)` sampling. Mirin's `mods[pn]` + additive ease
   stack (L992) is the same shape. (`channels.py`, `note_mods.py`).
2. beat->time up front as a clock. Matches `GetElapsedTimeFromBeat`.
3. One-shot funcs fired once + perframe/integrator as the dynamic tail
   (items 16/59). Matches Mirin `func`/`perframe` split exactly.
4. Item 52 note-path / chart-defined-mod grid-sampling design
   INDEPENDENTLY REINVENTED Mirin `definemod`. Our design is sound; use
   Mirin's definemod as the reference for the input->fn->output contract
   (single eased percent in, engine-mod deltas out).
5. Computed channels / dataflow (item 52, DESIGN axis 6) = Mirin `node`.
   Confirmed the "pure fn of channels = compilable, actor-poke = needs
   sampling" split.

WRONG / to correct:
1. **The approach-chase compiler is classic-only.** `channels.py`'s
   fapproach/clearall/float-back model is CORRECT for gat but WRONG for
   Mirin (which snaps `*-1` and eases in Lua). A Mirin front-end must
   bypass it. Do not reuse `mod_channels.py`'s window->approach path for
   Mirin. (Highest-value; see report.)
2. **Easing fidelity.** We map recorded tweens to the osu Keyframe enum
   (RecordingActor). Mirin's transient family (bounce/pop/pulse/spike)
   and combinators (flip/blendease) and parameterized elastic/back have
   NO osu id. For Mirin, evaluate the real Lua ease and bake dense
   breakpoints (or add a per-segment callable). Our `_Segments` linear
   interpolation would visibly wrong-shape any non-linear Mirin ease.
3. **item 31/158 receptor snap-back** is a classic-template revert
   artifact; Mirin has no such revert (the ease returns to 0 by its own
   curve). Don't carry that heuristic into the Mirin path.

MADE EASY BY MIRIN THAT WE BUILT THE HARD WAY (file pointers):
- Per-frame `clearall + reapply` decoding, window coalescing, and
  approach-revert modeling (`mod_channels.py` ~330 lines,
  `channels.py` ~220 lines, plus item 31 snap-back debugging) - Mirin
  charts hand us `eases` as clean keyframe pairs; NONE of that machinery
  runs. The whole "what speed does it revert at" saga
  (`mod_channels.py` docstring) is moot for Mirin.
- Per-column fan-out (`movex0..7`) we hand-mapped in
  `arrow_effects.py`; Mirin's `movex` definemod (L1256 repeat8) declares
  it, so a Mirin chart's per-column intent arrives pre-fanned.
- `xmod`->scroll-multiplier (item 16, `_scroll_mult_timeline`): Mirin's
  `xmod` definemod (L1274) writes the speed modstring itself - harvest
  its output rather than re-deriving.

---

## 4. Adoption plan (ordered)

Prereq: LuaHost luajit21 dialect must allow controlled `loadstring` +
`setfenv` (template.lua codegens nodes/funcs). Add a Mirin sandbox
profile in `analysis/player/render/lua/` alongside the fluXis one.

Step 1 - Vendor + run the template. Bring `template/*.lua` (or point at
the chart's bundled copy) into a scratch runnable. Stub the minimal host:
`GAMESTATE:GetCurrentSong` + `song:GetElapsedTimeFromBeat` (from our
parsed BPM map), `GAMESTATE:ApplyModifiers` (no-op for harvest),
`SCREENMAN`/`SCREEN*` + actor handles as recording stubs (reuse
`recording_actor.py`), `GAMESTATE:GetSongBeat`/`GetSecsIntoEffect` fed by
our clock during ticking. Confirm `ready_command` completes and `eases`/
`funcs`/`nodes` populate. Validate against `spec/*` semantics.

Step 2 - New module `analysis/games/notitg/mirin_compiler.py`:
    detect_mirin(chart_files) -> bool          # section 2.4 sniff
    compile_mirin(chart, host) -> dict         # same compiled-dict shape
      as modfile.compile_modfile: emits mod channels (keyframes, NO
      approach), field/scene channels, storyboard elements (actor pokes
      from actor-binding nodes), custom streams (item 51).
  - Harvest `eases`: for each row, evaluate `easefn` on a sample grid
    over `[start_time, start_time+len]` (density from a per-ease hint;
    linear/instant/set -> 2 points), emit breakpoints per (mod,player).
    Handle transient (`easefn(1)<0.5`) by returning to the pre-ease
    target. Feed a NEW keyframe channel builder (NOT the approach
    compiler) - or extend `_Segments` with a per-segment ease callable.
  - Harvest pure `nodes`/`definemod`: sample inputs, run fn, emit output
    channels (recursing until engine mods). Actor-poke nodes -> record
    via integrator into storyboard timelines.
  - Harvest `funcs`: one-shot funcs fire once at `start_time`; perframes
    -> `update_integrator.run_update_integration` over their span,
    intercepting `poptions`/`mods` writes as channel keyframes.
  - Subtract 1 from Mirin plr for our 0-based player index.

Step 3 - Wire into `NotitgAdapter`: in the adapter, detect dialect per
chart; route to `mirin_compiler.compile_mirin` vs the existing
`modfile.compile_modfile`; both return the same compiled dict consumed
by `note_mods()`/`storyboard()`/`effects()`. Zero renderer change - the
channel IR and note-path pipeline already consume this shape.

Step 4 - Easing library port. Port `ease.lua` to a small Python module
(pure functions, name->callable, incl. flip/blendease/with1param/
with2params and the transient family). Used by the sampler in step 2.
Cross-check numerically against running the Lua (LuaHost) on a grid -
Mirin's `eases_spec.lua` asserts exact values (e.g. `outExpo(.5)*10000`)
so it doubles as a parity oracle.

Step 5 - Test strategy. The repo SHIPS a runnable example chart
(`Song.sm` + `Song.ogg` + `lua/mods.lua`) and full `spec/*_spec.lua`
busted suites. Golden strategy:
  - Port key `spec` assertions (ease additive stacking, transient
    stick/return, plr split, set/acc, node fan-out) to
    `tests/test_mirin.py`, running our compiler against the same inputs
    and checking sampled channel values match the spec's expected
    numbers. The spec files are the oracle - Mirin's own busted tests.
  - Use `Song.sm` + a trivial `mods.lua` (`ease {0,1,outExpo,100,
    'invert'}`) as an end-to-end: compile -> sample invert channel ->
    assert it tracks `outExpo`.
  - Detection unit tests: a Mirin file, a classic gat snippet, a Cat
    snippet -> correct dialect routing.

Deferred: actor-poking nodes/perframes that touch our unsupported 3D /
shader tiers ride the existing Stage-B backlog; Government Knows (Cat
framework) is a separate third front-end, not part of Mirin adoption.
