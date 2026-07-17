# Compiled Document: the native modchart model

Consolidation target for the render/effects/mods systems. One object
per chart, emitted by every game adapter, consumed by the player
through one loading path. The serializable native map format (unified
cross-game maps, Rust port boundary) is a dump of this model; this
pass builds the model only, with zero behavior change (suite + oracle
baseline `refs/notitg/oracle_baseline` must stay stable).

## The six axes

1. **Layers** - compositing strata. Draw order is a per-node property;
   capture scopes (AFT/proxy sources, shader-pass inputs) are declared
   ranges over strata, not bespoke pixmaps.
2. **Groups** - membership over the superset of map objects: storyboard
   elements, the notefield, single notes / note subsets (per-note
   membership arrays, the Quaver SV-group pattern generalized), field
   copies, text. Nodes carry property timelines; children compose.
3. **3D object transforms** - translate/rotate/scale/skew natively in
   3D with one root projection (camera, fov, vanishing point). The
   QPainter homography approximation and a future GL executor are two
   backends of the same tree.
4. **Pixel projectors** - shaders scoped to a layer range (fullscreen
   post) or to a node (per-actor).
5. **Time = the integral engine** - every timeline is keyed
   `(clock_key, curve)`; clock keys come from the SV integral system
   (`analysis/player/sv/`): song time, beat = integral of bpm, scroll
   position = integral of sv, per-group engine keys. Rate mods, stops,
   warps, and NotITG effectclock all reduce to clock choice.
6. **Data streams** - named value/text channels: chart-written custom
   buffer entries and replay-derived gameplay tallies (combo, misses,
   judgment counts). Properties and text content bind to stream names.

Leaves are content: sprite, text, rect/poly, notefield, capture-of-node.
Notes are leaves whose transform additionally samples a note-path curve
by scroll offset (axis 5 applied to axis 3).

## Current embodiments to absorb (entry point -> axis)

- `EffectFrame.transform` / `scene_transform` / `SCENE_TOP_Z` band ->
  group-node transforms + layer strata (axes 2, 3, 1).
- `EffectFrame.draws` (z-ordered painters) -> leaves with layer slots.
- `EffectFrame.fields` (transform, opacity, scope) 3-tuples ->
  capture-of-node leaves with declared capture ranges (axes 1, 2).
- `EffectFrame.shaders` + ShaderStackEffect + chart shader bridge ->
  pixel projectors on layer ranges (axis 4).
- ctx stashes (`candidate_dx/_alpha/_rot_deg/_zoom`,
  `hold_body_samples`, `receptor_offsets`) -> note-path curve samples
  handed over in one shape (axes 3, 5); the batched `note_offsets`
  call is the analytic curve evaluator (implementation #1), grid-
  sampled surfaces for chart-defined mods come later (implementation
  #2).
- `_scroll_mult_timeline`, SV engine keys -> axis 5 clocks.
- Storyboard IR (`render/storyboard/model.py`) -> already a group tree
  of leaves; becomes the same node type as everything else.
- Palette / layer_fade / opacity -> node property timelines.
- Integrator streams (`update_integrator.py` dense keyframes,
  ApplyGameCommand windows, proxy-grid copies) -> ordinary node
  timelines; no special casing.
- Adapter design kwargs (`storyboard(design=..., fit=...,
  clip_design_box=...)`, `ref_space.py` REF_W/H) -> the document's
  design-space header via a new `GameAdapter.design_space()` surface;
  the player maps design->screen in exactly one place.

## Migration phases (each lands green + oracle-stable)

1. **Document skeleton + design space**: `CompiledDocument` dataclass
   (design space header, node list, stream table, clock table);
   `design_space()` adapter surface replacing scattered constants and
   kwargs. Adapters keep their current outputs; the document wraps
   them.
2. **Note-path extraction**: define the path interface; the analytic
   ArrowEffects pipeline becomes its first implementation; ctx stashes
   become path-sample handoffs (one shape). Consumers (heads, bodies,
   tails, receptors) read the path.
3. **Group/layer tree**: storyboard elements, field, copies, and effect
   draws become nodes with (group parent, layer slot); EffectFrame
   channels become property timelines on nodes. Fixed pipeline order
   becomes default strata.
4. **Capture ranges + projectors**: field/backdrop pixmaps and shader
   pass inputs declared as layer ranges; the two hand-built capture
   scopes become instances.
5. **Clocks + streams**: timelines gain clock keys (default: song
   time); data streams table with gameplay tallies precomputed and
   custom-buffer writes recorded at compile time; bitmaptext content
   binding.

Phases 1-2 are prerequisites for wave-4 features (dual-player fields
want the tree; chart-defined mods want the path interface; the
scoreboard wants streams). 3D-native transforms (axis 3) land with the
scene-projection work on the phase-3 tree.

## Invariants

- Per-game code compiles INTO the document; no game name appears in
  the player/render path.
- Every phase: full suite green, oracle structurally identical at
  t = 8, 42, 150, 236, 253, 322, 346, 383, 415 (baseline montages in
  `refs/notitg/oracle_baseline/`).
- Hot-path sampling stays vectorized / O(log n) per timeline; the
  document is data (lightest containers), not behavior.
