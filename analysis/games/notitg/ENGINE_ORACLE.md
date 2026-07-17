# NotITG Engine Oracle

Source-of-truth map for the NotITG rendering interface, built by reading the
engine directly rather than running it (user directive 2026-07-17: NotITG's
lineage is open source; read the source). Three parts:

1. API surface inventory - the actor/Lua function surface the engine exposes,
   which of those are NotITG fork additions vs OpenITG baseline, cross-referenced
   against every function `gat` actually calls, with our support status.
2. Rendering interface source map - how gat's calls flow at draw time
   (Player -> NoteField, ActorProxy, ActorFrameTexture, ActorFrame perspective,
   draworder, Actor effect oscillators), each with file:line and a one-line
   "what our engine must implement" note.
3. Trace-diff protocol - OPTIONAL future work (running the game is deprioritized).

## Sources

- Local install: `/mnt/Yucky/Rhythm Games/Players/NotITG/`, binary
  `Program/NotITG-v4.2.0.exe` (PE32 i386, SM 3.95 / OpenITG lineage, TaroNuke fork).
  No shipped Lua API docs (only `Docs/*.txt`: FAQ/ReleaseNotes/Licenses).
- Binary strings dump: `refs/notitg/exe_strings.txt` (33709 strings). The Lua
  binding registration names live here - this is the "engine tells us what to
  impl" surface.
- OpenITG source (the public ancestor, same renderer architecture):
  `refs/notitg/openitg-src/` (cloned from `openitg/openitg`). NotITG itself is
  closed-source; TaroNuke publishes only the theme.
- Local Etterna checkout `/home/yucky/etterna/src/Etterna/` - carries the SM
  lineage classes OpenITG lacks (ActorProxy, ActorFrameTexture) so their draw
  semantics can be read directly.
- gat pilot: `/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/UKSRT8/5. gat/`
  (lua/default.xml 5921 lines + bg/chara/fuck/modhelpers XML).

Fork-addition detection method: a symbol present in the NotITG binary strings
but absent from OpenITG source is a fork addition. Verified per-symbol with
`grep -rl <sym> openitg-src/src`.

---

## Part 1 - API surface inventory

### 1a. NotITG fork additions (in binary, ABSENT from OpenITG source)

These are the renderer capabilities NotITG added on top of SM 3.95 / OpenITG.
They are exactly the hard asks for our port.

| Symbol | Kind | Engine semantics | Our status |
|---|---|---|---|
| `ActorProxy` / `SetTarget` / `GetTarget` | actor class | Re-renders another actor's live draw under this actor's transform (see 2b). | field_instances.py models proxy-as-blit; live re-render approximated by sampling target timelines. |
| `ActorFrameTexture` (+ `Create`, `EnableDepthBuffer`, `EnablePreserveTexture`, `EnableAlphaBuffer`, `EnableFloat`, `GetTexture`, `SetTextureName`) | actor class | Renders its children into an FBO at its tree position (see 2c). | 'screen'/'full' field-capture scopes (aft_drivers.py, field_instances.py); depth/float buffers not modeled. |
| `SetShaderFlag` / `SetShaderFlagNum` / `GetShaderFlag` | GAMESTATE method | Chart-defined global shader toggles; semantics are per-chart (no published table). | shader_bridge.py maps a few keys -> screen_mirror/screen_tile frags; most keys chart-defined, skipped. gat's mod_shader all commented out. |
| `GetArrowShader` / `GetReceptorShader` / `GetHoldShader` / `GetArrowPathShader` / `ClearArrowShader` / `ClearReceptorShader` / `ClearHoldShader` / `ClearShader` | notefield shader binding | Per-notefield-element GLSL programs. | Not consumed (gat uses none live). |
| `GetRandomVanishTransform` | ActorFrame method | Randomized fov/vanish transform. | Not consumed. |
| `uniform1f`/`1i`/`2f`/`3f`/`4f` + `*fv` + `uniformMatrix2/3/4fv` | shader-program Lua setters | Custom uniform upload from Lua onto a `GetShader()` program. | notitg_compat.py sets custom float/vec uniforms via glUniformNf (item 104 trap: never setUniformValue(int,float)). Matrix uniforms not wired. |
| `AddDrawSplit` / `DrawExtraPixelsLeft/Right` / `DrawHoldHeadForTapsOnSameRow` | notefield draw controls | Fine-grained notefield draw splitting. | Not consumed. |
| `tan*` mod family (tandrunk/tantipsy/tantornado/tanbumpy(xyz)/tandigital/tanpulse/tanexpand/tanclip, + offset/period/size/spacing/speed companions) | modifiers | NotITG-extended per-note mods (cosecant variants). | arrow_effects.py covers the tan* family (memory item 37). |

Note: `SetVanishPoint`, `GetShader`, `luaeffect`, `GetEffectMagnitude`,
`customtexturerect` ARE present in OpenITG baseline (not fork-only) - but their
Lua exposure in NotITG is the surface charts use.

### 1b. gat call surface cross-reference (called-by-gat -> support -> semantics)

Every method gat invokes (counts from grepping the gat lua/bg XML). Status:
RECORDED (we harvest/compile it), CONSUMED (renderer acts on it),
PARTIAL, IGNORED.

| gat call (count) | Status | Engine semantics | Note |
|---|---|---|---|
| `x`/`y`/`z` (925/683/54), `addx`/`addy`/`addz` (34/25/1) | CONSUMED | Actor position set/accumulate in parent space. | recording_actor tween model. |
| `zoomto` (512), `zoom`/`zoomx`/`zoomy`/`zoomz` (29/16/6/3) | CONSUMED | Scale. | |
| `SetTextureFiltering` (514) | IGNORED | Texture min/mag filter. | Cosmetic; safe to ignore. |
| `linear`/`decelerate`/`accelerate`/`bounce*`/`spring`/`sleep` (tween verbs) | CONSUMED | Tween easings + delay. | Mapped to osu easing enum. |
| `hidden` (136) | CONSUMED | Visibility (own `hidden` channel, item 31). | |
| `finishtweening`/`stoptweening`/`stopeffect` (63/2/8) | CONSUMED | Tween/effect control. | |
| `rotationz`/`rotationx`/`rotationy` (82/35/23) | PARTIAL | 3D actor rotation. rotationz consumed 2D; rotationx/y -> field_3d homography for the field, but per-note/per-actor rot_x/y still largely 2D-approximated. | **GAP: true per-actor 3D.** |
| `skewx` (11) | PARTIAL | Skew shear. | field_3d handles field skew; general actor skew limited. |
| `basezoomx`/`basezoomy` (21/19) | CONSUMED | Base scale (basezoomy=-1 = y-flip, used by AFT copies). | |
| `diffusealpha`/`diffuse` (26/10) | CONSUMED | Alpha / color. | |
| `GetX`/`GetY`/`GetZ`/`GetZoom`/`getrotation`/`GetSecsIntoEffect` | RECORDED | Actor state readback (drives perframe copy transforms). | aft_drivers grid-samples driver closures vs compiled quad curves. |
| `GAMESTATE:ApplyGameCommand` (36) | CONSUMED | Per-frame mod injection ("mod,...") - the integrator's main input. | update_integrator.py, coalesced 158k->6k windows. |
| `GAMESTATE:GetSongBeat` (13) | CONSUMED | Current song beat (drives perframe math). | integrator song-time ticks. |
| `Broadcast`/`queuecommand`/`playcommand`/`queuemessage` (29/27/15/2) | CONSUMED | Message dispatch. | mod_stubs message dispatch (item 22). |
| `GetChild`/`GetTopScreen`/`SetTarget`/`Create` (18/52/28/2) | PARTIAL | Actor tree navigation + proxy/AFT target binding. | Tree splice + named binding. |
| `SetTexture`/`GetTexture`/`SetTextureName` (5/5/2) | PARTIAL | AFT render-target texture handoff. | Capture-scope path. |
| `effectmagnitude`/`effectperiod`/`effectclock`/`vibrate`/`wag`/`bob`/`bounce`/`spin`/`effectm` (20/6/7/9/3/2/2/-/1) | **IGNORED** | Actor **effect oscillators** - continuous sinusoidal/random self-animation (see 2e). Recorded as static values, NOT re-synthesized as oscillators. | **GAP #3 - see below.** |
| `SetVanishPoint` (1) | **IGNORED** | Sets fov vanish center for the actor's perspective (see 2d). | **Not recorded** in recording_actor. **GAP #1.** |
| `GAMESTATE:SetShaderFlag`/`SetShaderFlagNum` (2/2) | PARTIAL | Chart shader toggles. | Bridge maps some keys; gat's are commented out. |
| `SetShaderFlag`/`SetShaderFlagNum` via GAMESTATE | PARTIAL | See 1a. | |
| `EnableDepthBuffer`/`EnableAlphaBuffer`/`EnableFloat`/`EnablePreserveTexture` (2 each) | PARTIAL/IGNORED | AFT buffer config. PreserveTexture = accumulate (feedback). Depth/float unmodeled. | 'screen' scope gives feedback; depth buffer (3D z-test in captures) IGNORED. |
| `settext`/`GetText`/`SetWidth`/`SetHeight` | PARTIAL | Text + size. | bitmaptext element. |
| `SetShaderFlag`, `KeyPress`, `IsPlayerEnabled`, `GetPreference`/`SetPreference`, `GetDisplayWidth`/`Height`, `GetCurStageStats`, `SystemMessage`, `GetCurrentSong`, `GetVendor`, `setstate`, `animate` | MIXED | Misc engine queries / state. | Most stubbed as constants; `setstate`/`animate` = sprite frame (sprite_sheet.py). |

### 1c. Modifier vocabulary (full engine set, from binary)

The binary contains the complete `PlayerOptions::FromString` mod token set. Our
`arrow_effects.py` covers the per-note x/y/z/rot/alpha/zoom mods (memory item 37
= completeness landed). Confirmed present in binary and covered: boost/brake/
wave/expand/boomerang, drunk(z)/dizzy/confusion/tornado(z)/tipsy/bumpy(z)/beat(z),
invert/flip/reverse/split/alternate/cross/centered, hidden/sudden/stealth/blink/
dark/blind, mini/tiny, digital/zigzag/sawtooth/square/bounce warp family, pulse/
attenuate/parabola/xmode, movex/movey/movez, confusionoffset, hallway, and the
entire `tan*` family. Perspective family (`space`/`distant`) + `randomvanish` are
the scene-projection frontier (part 2d).

---

## Part 2 - Rendering interface source map

Reference chain at draw time: `ScreenGameplay` (an ActorFrame) -> `m_Player[p]`
(a Player, itself an ActorFrame) -> `m_pNoteField` + `m_Judgment` + `m_Combo`
children. Everything is an Actor; the whole scene is one actor tree drawn
depth-first. gat overlays its modfile actors as additional ScreenGameplay
children spliced via #FGCHANGES.

### 2a. Player is an ActorFrame; NoteField/Judgment/Combo are its children

`openitg-src/src/Player.cpp:96-104` - `AddChild(&m_Judgment); AddChild(&m_Combo);
AddChild(&m_HoldJudgment[c]); m_pNoteField = new NoteField;`.
`ScreenGameplay.cpp:246-248` - `m_Player[p].SetName("PlayerP%i"); SetXY(fPlayerX,
SCREEN_CENTER_Y); AddChild(&m_Player[p])`.

- What our engine must implement: an ActorProxy targeting the Player copies the
  notefield AND the judgment/combo actors (memory item 54). P1/P2 X placement is
  `fPlayerX` (theme metric ScreenGameplay PlayerP#X) with Y = SCREEN_CENTER_Y=240;
  gat repositions via pokes. Confirms dual-field capture must optionally include
  per-side judgment/combo (item 43/50/54).

### 2b. ActorProxy::DrawPrimitives - re-render, not texture copy

`etterna/src/Etterna/Actor/Base/ActorProxy.cpp:18-27`:
```
void ActorProxy::DrawPrimitives() {
  if (m_pActorTarget != nullptr) {
    bool bVisible = m_pActorTarget->GetVisible();
    m_pActorTarget->SetVisible(true);
    m_pActorTarget->Draw();               // re-runs the target's whole Draw()
    m_pActorTarget->SetVisible(bVisible);
  }
}
```
The proxy sits at its own position in the tree, so `m_pActorTarget->Draw()` runs
inside the proxy's already-pushed transform (ActorFrame pushes its matrix before
drawing children). It force-shows the target even if hidden, then restores.

- What our engine must implement: a proxy is the target re-evaluated this frame
  under the proxy's transform - so P1p..P6p copies show the SAME live notefield
  state (same note positions, same judgment) transformed differently, never a
  stale snapshot. Our blit-of-capture is a valid approximation ONLY because the
  target and copies share the same per-frame field content; where a proxy targets
  a DIFFERENTLY-modded field (per-player mods), the copy must re-evaluate that
  side's note pipeline, not blit P1's pixels. This is the exact seam behind items
  43 (dual fields) and 32 (per-copy capture source).

### 2c. ActorFrameTexture::DrawPrimitives - FBO capture at tree position

`etterna/src/Etterna/Actor/Base/ActorFrameTexture.cpp:79-88`:
```
void ActorFrameTexture::DrawPrimitives() {
  if (m_pRenderTarget == nullptr) return;
  m_pRenderTarget->BeginRenderingTo(m_bPreserveTexture);
  ActorFrame::DrawPrimitives();          // draw children into the FBO
  m_pRenderTarget->FinishRenderingTo();
}
```
Create (`:42-75`): allocates an FBO sized `m_size.x/y` (rounded), with optional
depth/alpha/float buffers, registered under a texture name other actors sample.

- Frame ordering / one-frame delay: the capture happens WHEN the AFT node is
  reached in depth-first draw order. A Sprite that samples the AFT texture and is
  drawn LATER in the same frame sees THIS frame's content (no delay). A sampler
  drawn BEFORE the AFT (or the AFT sampling ITSELF via `PreserveTexture`) sees the
  PREVIOUS frame - that is the feedback/echo trail case. `PreserveTexture=true`
  means "don't clear the FBO", i.e. accumulate across frames (gat's echo/datamosh).
- What our engine must implement: our 'screen' scope (item 84) already models the
  one-frame-delayed full-composite feedback (basezoomy=-1 y-flip = GL texture
  origin). The unmodeled parts: (i) depth buffer in captures (3D z-tested scenes,
  Government Knows), (ii) arbitrary AFT sizes/names as first-class render targets
  bound to arbitrary sampler sprites (CaptureRange in the consolidation doc).

### 2d. ActorFrame perspective - fov + vanish point application point

`openitg-src/src/ActorFrame.cpp:165-219` (`DrawPrimitives`):
```
if (m_fFOV != -1) {
  DISPLAY->CameraPushMatrix();
  DISPLAY->LoadMenuPerspective(m_fFOV, m_fVanishX, m_fVanishY);   // :172
}
... draw children (SortByZPosition if m_bDrawByZPosition) ...     // :194-205
if (m_fFOV != -1) DISPLAY->CameraPopMatrix();                     // :216-219
```
Defaults `:33-35`: `m_fFOV=-1` (off), `m_fVanishX/Y = SCREEN_CENTER_X/Y`.
`SetVanishPoint` (NotITG Lua) writes `m_fVanishX/Y`; fov set via `fov`/XML `FOV`.

- Application point: perspective is a CAMERA matrix pushed around the frame's
  children, with the vanish point as the projection center. So fov on a group
  projects the whole subtree (notefield + copies + sprites) through one vanishing
  point. This is a true 4x4 perspective, NOT a per-actor 2D skew.
- What our engine must implement: our transform3d.py (item 68b) builds the 4x4
  SM-semantics stack + fov/vanish projection -> planar homography; field_3d.py
  (item 89) consumes P1 rot_x/y+skew through it. The MISSING input is
  `SetVanishPoint` recording (item 89 follow-up, GAP #1) - without it every fov
  scene uses default center 320,240 and off-center perspective scenes are wrong.
  gat calls SetVanishPoint once; Government Knows uses fov+vanish heavily.

### 2e. Actor effect oscillators - Actor::UpdateInternal / UpdateEffect

`openitg-src/src/Actor.cpp`. Effect state advances every frame by an effect
clock (`:559-606`), then the oscillator writes into `m_tempState` (`:248-365`):

- Clock (`:564-590`): `m_EffectClock` in {CLOCK_TIMER (secs), CLOCK_BGM_BEAT,
  CLOCK_BGM_TIME (music), CLOCK_LIGHT_*}. `effectclock,music` in gat = beat/time
  synced oscillation. `SetEffectClockString` `:720-731`.
- Period/phase (`:271-277`): `fSecsIntoPeriod = fmodfp(secsIntoEffect + offset,
  period + delay); pct = SCALE(fSecsIntoPeriod, 0, period, 0, 1)`.
- Oscillators (`:290-365`), each `= f(pct) * m_vEffectMagnitude`:
  - `diffuse_blink`/`diffuse_shift`/`glow_blink`/`glow_shift`/`rainbow` (color).
  - `wag` `:331` = `rotation += magnitude * sin(pct*2pi)`.
  - `spin` `:334` / `:598` = `rotation += effectDelta * magnitude` (continuous).
  - `vibrate` `:337-340` = `pos += magnitude * randomf(-1,1) * zoom` (per-axis).
  - `bounce` `:342-345` / `bob` `:351-354` = `pos += magnitude * fPercentOffset`
    (bounce = abs sine, bob = sine).
  - `pulse` `:360-362` = zoom lerp between magnitude[0] and magnitude[1].
- `effect_lua` (`:248`, `SetEffectLua` `:763`) = arbitrary Lua per-frame effect.

- What our engine must implement (GAP #3): these are CONTINUOUS self-animations
  parameterized by (magnitude, period, offset, clock), NOT keyframes. gat uses
  vibrate/wag/bob/bounce/spin + effectmagnitude/period/clock (memory item 53 #9
  flagged "NEW medium"). recording_actor.py records the static setters but does
  NOT re-synthesize the oscillation, so any actor whose motion IS the oscillator
  (screen-shake vibrate, wagging arrows, bobbing chars) renders frozen. Fix shape:
  record (effect kind, magnitude vec, period, offset, clock) and emit an analytic
  oscillator channel sampled at playback t (the compiled-document model already
  samples curves per frame; this is one more curve source, deterministic since
  clock is song-time/beat - the randomf in `vibrate` needs a seeded per-actor RNG
  like the fuck-pool spawner, item 89).

### 2f. NoteField::DrawPrimitives - receptors first, draw-range expansion

`openitg-src/src/NoteField.cpp:429-458`:
- `:436-437` - `cur->m_ReceptorArrowRow.Draw()` draws receptors FIRST, then notes.
- `:453-455` - draw-range scale: `fDrawScale = (1 + 0.5*|PerspectiveTilt|) *
  (1 + |MINI|)`; first/last pixel to draw multiplied by it. Confirms tilt AND mini
  widen the culling window (relevant to items 80/87 - mini is a notefield-level
  zoom, no per-column spacing term; item 91 verdict).
- `:447-450` - boomerang+centered draws earlier (culling peak) - the deferred
  boomerang visibility contract (item 37).
- Per-note position pipeline (`:211-338` region + ArrowEffects): `GetYOffset ->
  GetYPos -> GetXPos -> GetZPos -> GetRotationX/Y/Z`. GetRotationX (`ArrowEffects
  .cpp:344`) = roll * yOffset/2; GetRotationY (`:354`) = twirl * yOffset/2;
  GetRotationZ (`:364`) = dizzy phase. These are the ONLY per-note 3D rotations in
  OpenITG; NotITG adds actual rotationx/y actor pokes + confusionx/y projected
  through fov (item 82 per-note 3D).

- What our engine must implement: receptors on the same path as notes drawn first
  (item 14, done). The `fDrawScale` tilt/mini culling expansion must be honored so
  tilted/mini fields don't pop notes at edges. Per-note z (`GetZPos` `:522`) and
  brightness/zoom (`GetBrightness` `:507`, `GetZoom` `:543`) feed the perspective
  scale - item 82's real-z path retires our z->zoom proxy.

### 2g. draworder / z-sorting

`openitg-src/src/ActorFrame.cpp:194-205` - if `m_bDrawByZPosition`,
`ActorUtil::SortByZPosition(subs)` before drawing; else tree order. The binary's
`draworder` string is the per-actor draw-order slot (SM `SetDrawOrder`), a
mutable integer that re-slots an actor within its parent's draw sequence.

- What our engine must implement: draw order is (i) tree order by default, or
  (ii) z-position sort when the frame opts in, or (iii) explicit per-actor
  `draworder` slot. Our fixed pipeline + z-band split approximates this; the
  consolidation layer model (item 58b) makes draw-order a mutable per-node
  property - matches the engine.

---

## Part 3 - Trace-diff protocol (OPTIONAL future work)

Running the game is deprioritized (source reading supersedes it). If a follow-up
ever wants a behavioral oracle, the shape is below. wine 9.x + Proton 10 + xvfb
are installed on this host (`/usr/bin/wine`, `~/.steam/.../Proton 10.0`,
`/usr/bin/xvfb-run`), so a headless run is feasible but unattempted.

### 3a. Capture (if ever run)

- Copy gat to a scratch song dir (never touch the original). Add a logger actor
  to the copy's `lua/` whose `UpdateCommand`/`func` dumps per frame: song time t,
  P1/P2 field x/y/rotationx/y/z/zoom, each active proxy's GetX/GetY/getrotation,
  and a few `GetEffect*`/mod readbacks, to a CSV via Lua `io`. NotITG's Lua
  sandbox `io` availability must be checked against the classic template first
  (the template itself writes files, so io.open is likely permitted).
- Invocation: `xvfb-run -s "-screen 0 1280x720x24" wine
  "Program/NotITG-v4.2.0.exe"` with autoplay + the copied chart, or Proton via a
  compatdata prefix. GL under Xvfb needs llvmpipe/`LIBGL_ALWAYS_SOFTWARE=1`;
  expect this to be the failure point (the engine wants real GL).
- Output target: `refs/notitg/engine_trace_gat.csv`, one section (e.g. t=37-80).

### 3b. Diff against our compiled timelines

- Map trace columns -> compiled dict streams:
  - trace `field.rotationx/y/z, x, y, zoom` <-> compiled field/scene transform
    channels (field_3d.py output + fields channel).
  - trace `proxy_i.x/y/rot` <-> compiled `field_copies[i]` transforms
    (modfile `_proxy_grid_copies` / `_SumTimeline`).
  - trace per-note (col, y) sample <-> `note_offsets` over candidates at t.
  - trace `GetEffect*`/mod readback <-> `ModChannels.values_at(t)`.
- Alignment: both clocks are SONG TIME. The compiled document is authored in
  chart time; the trace logs t = engine song second. No cross-correlation needed
  (unlike the video oracle's +19.8s offset) - the logger emits the same clock the
  compiler uses. Sample both at a shared t grid (e.g. 60 Hz over the section).
- Tolerance: positions to ~1 engine px (SM px, design 640x480); rotations to
  ~0.5 deg; alpha/zoom to ~1%. Waveform/oscillator channels compared on VALUE at
  matched t (not shape-correlated) since phase is deterministic from song time.
  Flag any channel whose max abs error over the section exceeds tolerance ->
  ranked engine-vs-compile regression report. This becomes a golden oracle
  generator: cache the trace CSV per (chart, section) and re-diff on every
  compiler change (the parity harness `tests/test_engine_parity.py` already does
  this for the pure ArrowEffects math using a transliterated reference; a trace
  extends it to the actor/proxy/field layer that can't be transliterated).

---

## Top 3 implementation gaps the cross-reference exposes

1. **SetVanishPoint not recorded (GAP #1)** - the fov vanish center. transform3d
   + field_3d exist but always project through default center (320,240). gat calls
   it once; Government Knows depends on it heavily. Fix: record it in
   recording_actor and thread it into the field_3d/scene projection.
2. **True per-actor/per-note 3D rotation (GAP #2)** - rotationx/rotationy pokes are
   recorded but the field-level homography (field_3d) is the only consumer; per-
   note and per-non-field-actor 3D is still 2D-approximated. Engine projects every
   quad through the ActorFrame fov camera (2d/2f). Item 82's per-note 3D path.
3. **Effect oscillators frozen (GAP #3)** - vibrate/wag/bob/bounce/spin +
   effectmagnitude/period/clock are recorded as static values, never re-synthesized
   as continuous oscillators (Actor::UpdateInternal, 2e). Any actor whose motion IS
   the oscillator renders motionless. Deterministic to compile (clock = song
   time/beat; vibrate needs a seeded RNG).
