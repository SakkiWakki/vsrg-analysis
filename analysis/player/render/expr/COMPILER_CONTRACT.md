# Script-compiler ↔ timeline backend request contract

Status: DESIGN (spec, not yet implemented). This is the seam a future Rust
compiler base emits into, and it is deliberately **language- and game-neutral**:
the compiler core knows how to lower a script AST to this contract, nothing about
any specific game OR scripting language.

## Why this exists

VSRGs script their charts in different ways. Some use Lua (NotITG, Qwilight,
soundsphere, ...); some use their OWN format (BMS: `#RANDOM`/`#IF` control flow +
channel note/BGA events - NOT Lua). Rather than a per-game interpreter each,
there is ONE compiler base whose TARGET is universal: it lowers a script to the
timeline requests below, against the unified integrated timeline
(`analysis/player/render/scheduler.py` - Clock / Channel / EventSchedule, the SV
time-integral model).

The contract (the bottom seam) is the true invariant: EVERY scripting format -
Lua, BMS-format, future - compiles TO it. Languages differ at the front (each
needs its own parser/AST); games differ in vocabulary (verb/actor names); but
once lowered, they all become the same set of timeline requests. NotITG (Lua) is
the first proving ground and the validating example throughout; its measured
usage shows the contract covers a real game, it does NOT define the contract.

Two formats bound the contract from opposite ends, which is why both matter for
the design:
- NotITG (Lua modchart) exercises the FULL stack: continuous curves, a live
  per-tick tick-loop reading actor state, live queries. The "heavy" capability.
- BMS is DECLARATIVE, fully resolved at load/compile time (ref:
  hitkey.nekokan.dyndns.info/cmds.htm). Its only control flow is compile-time
  BRANCH SELECTION (#RANDOM/#IF/#SWITCH/#CASE - pick a pattern variant, no
  runtime eval). It reads NO live state, has NO per-frame anything. Events are
  the channel/measure model (#XXXYY:data = measure XXX, channel YY, data
  quantized across the measure; BPM #xxx03, measure-length/scroll #xxx02, notes,
  BGA). So BMS uses ONLY `EMIT_EVENT` over clocks + the EventSchedule backbone -
  none of curves/residue-tick/live-query.

IMPLICATION (important): the residue tick-loop and `QUERY_LIVE` are a
MODCHART-CLASS capability (Lua games with live per-frame drivers), NOT universal.
The truly universal core is narrower: `EMIT_EVENT` + `EMIT_CURVE` over clocks +
the EventSchedule/Channel timeline. A declarative format (BMS) compiles using
only that, its front-end being parse + a pure #RANDOM branch resolver that emits
events - zero interpreter involvement. The tick-loop/live-query is one OPTIONAL
capability a modchart adapter opts into.

The backend is still evolving (shaders, 3D, BMS, ...). The compiler depends only
on this REQUEST CONTRACT, which is stable; the backend grows underneath it.

## The four layers

```
   game script   (Lua | BMS control-format | future language)
        |
   [ per-LANGUAGE FRONT-END ]  parse -> that language's AST
        |                       (Lua games share one; BMS needs its own parser;
        |                        different syntax, same downstream)
        v
   [ per-GAME VOCAB ADAPTER ]  map this game's names/verbs -> contract kinds
        |                       (NotITG: getaux/ApplyModifiers/GetSongBeat ->
        |                        clock / live-query / event; other games: theirs)
        v
   [ COMPILER CORE ]           language-neutral: lower AST -> requests, split
        |                       analytic vs residue, run the residue tick loop
        v
   [ TIMELINE REQUESTS ]       <-- THIS CONTRACT: what the core emits
        |
   [ integrated timeline backend ]   Clock / Channel / EventSchedule
```

Note the split of "adapter" into two: LANGUAGE front-end (parsing - BMS forces
this to be separate, since its format is not Lua) and GAME vocab mapping. A Lua
game reuses the shared Lua front-end and supplies only a vocab adapter; BMS
supplies both a front-end and a vocab adapter. The core below is untouched by
either.

## The request vocabulary (what the core emits)

The core lowers a body into exactly these. Everything a game's Lua expresses
must reduce to one of them; anything that cannot is `UNMODELED` (fallback,
never guessed).

### 1. `EMIT_CURVE(target, prop, curve, clock)`

A property that evolves continuously as a function of a clock. The analytic
path (measured 56% of NotITG setter pokes) produces these directly - no
sampling. `curve` is an expression over the clock coordinate + constants +
native math; `clock` names a timeline Clock (song-time / beat integral / SV
integral / effect-timer loop).

- NotITG example: `p:x(320 + 40*math.sin(beat))`
  -> `EMIT_CURVE(p, x, |c| 320 + 40*sin(c), clock=beat)`
- The rate-mod / stop / warp reductions are CLOCK CHOICE, not curve changes
  (`scheduler.py`): "oscillate every 2 beats" vs "every 2 seconds" = same curve,
  different clock.

### 2. `EMIT_EVENT(when_clock, when_value, event)`

A discrete thing at a point on a clock: a mod window opening, a message
dispatch, a tween start, a one-shot state change. Feeds `EventSchedule`.

- NotITG example: `GAMESTATE:ApplyModifiers('...')` inside a beat gate
  -> `EMIT_EVENT(clock=beat, value=<gate start>, MOD_WINDOW(...))`.
  (measured: ApplyGameCommand 2599 + ApplyModifiers 1648 - these are EVENTS,
  not method crossings.)

### 3. `EMIT_SAMPLED(target, prop, keyframes, clock)`

The residue path (measured 44%): a property whose value each tick depends on
LIVE state the compiler cannot characterize analytically (another actor's
current position, an accumulator). The core RUNS the residue over the tick grid
NATIVELY (the tick loop is Rust, permanently - this is the load-speed motivation:
parsing+execution must be native) and emits the resulting piecewise curve as a
Channel over its clock. Still a timeline Channel, still clock-named - just
sampled rather than closed-form. Only declarative formats (BMS) never reach here.

### 4. `QUERY_LIVE(handle, kind) -> value`  (per-tick backend call = the migration frontier)

During the residue tick loop, the core needs the CURRENT value of something the
backend owns. It is a NARROW, TYPED value query over an opaque `handle` (an
actor id) - NOT a generic method dispatch. The adapter resolves game vocab to a
`kind` at compile time.

This trait (QUERY_LIVE + poke + RESOLVE_HANDLE) is the MIGRATION FRONTIER: it is
drawn wherever Python currently stops and Rust begins. TODAY its implementor is
PYTHON (the Surface/SimActor, not yet ported - fine while adapters/backend are
unfinalized), so the Rust tick loop bridges out to Python here. As each piece
stabilizes and ports, the SAME trait gets a Rust implementor and the crossing
becomes native - the core never changes. The frontier moves inward until it
vanishes (all Rust). Design it as one clean trait now = every port is a drop-in.

- Measured NotITG per-tick reads (the whole set, ~6 kinds = 95%): a numeric
  property (getaux 992 / GetX 893 / GetY 309 / GetZ 110 / GetShader 231) and
  text (GetText 126). Tail is single/double digit.
- Contract kinds are GENERIC: `VALUE(prop)`, `TEXT`. The adapter maps a game's
  getter names onto them. `getaux` is a NotITG prop like any other here.

### 5. `RESOLVE_HANDLE(scope, name) -> handle`  (mostly compile/setup, not per-tick)

Topology: turn a name into a backend handle (an actor / child / proxy target).
NotITG `GetChild` (7787) is this - overwhelmingly setup, not a per-tick value.

## What NEVER becomes a request (dissolved at compile time)

The adapter resolves these to the backend's native primitives; they are not
crossings and not runtime work. This is the "use the backend's power, don't
reproduce the Lua path" rule:

- clock reads (`GetSongBeat` 2242 / `GetSongTime` 174) -> a timeline Clock. The
  core already holds the clock; never a query.
- native math (`math.sin/cos/min/...`) -> native, inlined into a curve.
- constants / no-ops (`GetTopScreen` 6021 as a handle constant; `SystemMessage`,
  `PlayOnce`, `IsEditMode`, `GetVersionDate`) -> folded away.

## Rust crate internal shape (core = router; language piece = thin)

The crate is two parts, deliberately (user): NOT rebuilding a CFG parser.

- NEUTRAL CORE = the ROUTER + timeline-emission engines. It does NOT parse and
  knows no language. It receives normalized, adapter-tagged nodes and ROUTES
  each to the right machinery: analytic -> curve emitter, event -> schedule,
  residue -> native tick loop, live-read -> QUERY_LIVE. It also owns the
  analytic-vs-residue CLASSIFICATION (structural: does this expr read only
  clock+math+const?) - classification is routing, so it is core. It owns the
  value model, scope/store, data tables, tick loop. This is `frame_compile_exec`
  generalized: a node->request router instead of node->Python-closure.
- LANGUAGE PIECE (per language, NOT a from-scratch parser) = produces / accepts
  that language's AST and its value semantics (Lua truthiness/and-or), then
  hands normalized nodes to the core. Lua reuses the existing parser (Python
  today); a language piece is thin - shape + semantics, not reprocessing.
- ADAPTER (per game) = THIN pass-through: supplies vocab facts (which names are
  clocks / live-queries / events) and passes the node to the relevant core
  routing. Not a processing layer - a routing-hint provider.

Dataflow: parse (language piece) -> adapter tags vocab -> core routes to emitter.

## Core-owned state (no backend involvement)

The core owns, natively, everything the residue reads that is NOT live-backend:
- the scope/global store (accumulators the body itself writes: `gat_frame`, ...),
- read-only DATA TABLES the load pass built (`mods`/`v`/`e`/... - measured: read
  millions of times, 0 hash-keyed constructors corpus-wide, so a growable array
  / `Vec<Value>` suffices, NOT a Lua table object model),
- the value model: number / string / bool / nil / array / func / UNRESOLVED,
  with operand semantics per the source language (Lua `and`/`or` return-operand
  rules for Lua front-ends) and the UNRESOLVED "skip, don't guess" discipline
  (see `frame_eval.py`, the reference oracle). Value-model semantics that are
  language-specific (Lua truthiness) live with the front-end, not the core's
  request contract.

## Front-end + adapter responsibilities (per language / per game)

The ONLY non-core code:

LANGUAGE FRONT-END (per scripting language): parse the script to an AST the core
lowers. Lua games share one front-end (`expr/lexer.py` + `parser.py`); BMS needs
its own (its `#RANDOM`/`#IF`/channel format is not Lua). The AST node set the
core lowers is the shared target - a new language emits the same node kinds.

GAME VOCAB ADAPTER (per game): map the game's vocabulary onto the contract -
1. extract the per-frame body(ies) / event stream from the parsed script,
2. classify names/verbs: which are clocks (-> dissolved), which are live queries
   (-> QUERY_LIVE kinds), which are effects (-> EMIT_EVENT), which are no-ops;
   which setter verbs write which timeline property,
3. supply `RESOLVE_HANDLE` and the live `QUERY_LIVE` implementation over the
   game's actual engine state.

NotITG's vocab adapter is `guard_surface.py` + the sim today (Lua front-end
shared). A future Qwilight / soundsphere adapter reuses the Lua front-end +
supplies its own vocab. BMS supplies BOTH a front-end and a vocab adapter.

## Verification

The Python `frame_eval` interpreter is the REFERENCE ORACLE. Any compiler
implementation (Python analytic path, Rust core) is gated by `keyframe_diff`
(compare played-back timeline values): compiled == oracle == the game's own Lua
runtime, to tolerance. A construct outside the modeled subset is `UNMODELED` and
falls back - a strict-coverage floor, never a wrong guess.

## Open questions (resolve as games are added)

- Exact `QUERY_LIVE` kind set once a 2nd Lua game (Qwilight/soundsphere) is
  mapped - does the ~6 NotITG props generalize, or does each game add kinds?
  (modchart-class only; BMS never queries.)
- `EMIT_EVENT` taxonomy: NotITG needs {mod-window, message, tween-start}; BMS
  needs {note, bpm-change, measure-length/scroll, bga, bgm} at measure-fraction
  positions. Confirm one event vocabulary covers both, or that event KIND is an
  adapter-supplied payload the core treats opaquely (likely the latter - the
  core schedules "an event at clock coord C"; what the event MEANS is the
  backend's).
- `#RANDOM`/`#SWITCH` compile-time branch selection (BMS) vs Lua `if`: both are
  the core's control flow, but BMS resolves branches ONCE at load (a seed picks
  the variant) while Lua evaluates per context. Confirm the core's branch
  handling covers "resolve once with a seed" as a mode.
- Clock taxonomy completeness (song-time / beat / SV / effect-timer) vs a game
  that tells time a way these do not cover. BMS measure-length (#xxx02) is a
  scroll/SV-integral input - confirm it maps to the SV clock.
