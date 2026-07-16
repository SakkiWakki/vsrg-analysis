"""Per-layer eased alpha from `.ffx` layerfade events.

Ports fluXis's LayerFadeEvent: each event fades one FadeLayer's drawable
to `alpha` over `duration` with `ease`, holding after. fluXis uses
absolute transform sequences (BeginAbsoluteSequence + FadeTo), so later
events on the same layer win mid-fade -- exactly EventTimeline's
last-keyframe-wins sampling. Before any event a layer sits at full
opacity.

FadeLayer -> our layer names (via the layer registry):
- HitObjects -> every note-type leaf (taps/lns/mines/lifts/fakes/
  miss_holds/ghost_taps); fluXis fades playfield.HitManager, which holds
  all hit objects.
- Stage      -> 'lanes' (the stage frame / lane columns).
- Receptors  -> 'judgment' (the receptor / judgment line).
- Playfield  -> the whole field: note-type leaves + lanes + judgment.
- HUD        -> no field layer. fluXis fades a separate HUDAlpha; our HUD
  renders through a cached pixmap outside the field opacity path, so HUD
  fades are intentionally dropped (the field stays untouched).

The legacy `.ffx` keys `hitfade` and `playfieldfade` are LayerFadeEvent
aliases fluXis merges at deserialize time: `hitfade` keeps each event's
own layer (default HitObjects), `playfieldfade` forces layer Playfield.
Raw dicts still using them are merged here the same way.
"""
from __future__ import annotations

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import (
    EventTimeline, keyframes_from_events)

FADE_HITOBJECTS = 0
FADE_STAGE = 1
FADE_RECEPTORS = 2
FADE_PLAYFIELD = 3
FADE_HUD = 4

_NOTE_LAYERS = ('taps', 'lns', 'mines', 'lifts', 'fakes',
                'miss_holds', 'ghost_taps')
_STAGE_LAYERS = ('lanes',)
_RECEPTOR_LAYERS = ('judgment',)
_FIELD_LAYERS = _NOTE_LAYERS + _STAGE_LAYERS + _RECEPTOR_LAYERS

_LAYER_TARGETS = {
    FADE_HITOBJECTS: _NOTE_LAYERS,
    FADE_STAGE: _STAGE_LAYERS,
    FADE_RECEPTORS: _RECEPTOR_LAYERS,
    FADE_PLAYFIELD: _FIELD_LAYERS,
    FADE_HUD: (),
}

_REST = (1.0,)


def _merge_legacy(streams):
    """LayerFadeEvent list from `layerfade` plus the legacy aliases,
    matching MapEvents' deserialize-time concat order."""
    events = list(streams.get('layerfade') or [])
    events.extend(streams.get('hitfade') or [])
    for event in streams.get('playfieldfade') or []:
        if isinstance(event, dict):
            event = {**event, 'layer': FADE_PLAYFIELD}
            events.append(event)
    return events


class LayerFadeEffect:
    """Samples one EventTimeline per FadeLayer used and writes a dict of
    our-layer-name -> alpha onto `ctx.layer_opacities` for the renderer to
    apply. Never returns draws or a transform."""

    def __init__(self, streams):
        by_layer = {}
        for event in _merge_legacy(streams or {}):
            if isinstance(event, dict):
                by_layer.setdefault(int(event.get('layer', FADE_HITOBJECTS)),
                                    []).append(event)
        self._timelines = tuple(
            (layer, EventTimeline(
                keyframes_from_events(events, ('alpha',), (1.0,)),
                rest=_REST))
            for layer, events in by_layer.items()
            if _LAYER_TARGETS.get(layer))

    def __bool__(self):
        return bool(self._timelines)

    def at(self, ctx) -> EffectFrame | None:
        opacities = {}
        for layer, timeline in self._timelines:
            (alpha,) = timeline.sample(ctx.t_now)
            alpha = max(0.0, min(1.0, alpha))
            for name in _LAYER_TARGETS[layer]:
                opacities[name] = min(opacities.get(name, 1.0), alpha)
        ctx.layer_opacities = opacities or None
        return None
