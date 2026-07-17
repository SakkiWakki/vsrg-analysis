"""fluXis note color theming: the accent trio mapped onto lane columns,
animated by `colorfade` events.

fluXis colors notes from THREE accent slots -- Primary, Secondary,
Middle -- not one color per lane. Its `Theme.GetLaneColorIndex(lane,
keyCount)` assigns each lane one of those three slots per keymode
(symmetric outer/inner/centre patterns); `ColorManager` holds the live
slot colors, seeding them from the map's `colors` object (falling back
to theme defaults) and easing them at runtime via `ColorFadeEvent`s.

We reproduce that here: `note_palette()` resolves a static per-column
palette from the chart colors, and -- when the effect file carries
`colorfade` events for the main playfield -- wraps each faded slot in an
`EventTimeline` so the palette is sampleable at any time.

Ported from TeamFluXis/fluXis:
  Theme.GetLaneColorIndex / GetLaneColor (lane->slot map + defaults),
  ColorManager (slot seeding + defaults),
  ColorFadeEvent.Apply (per-slot TransformTo, gated by fade-* flags).
"""
from __future__ import annotations

from analysis.player.render.effects.timeline import EventTimeline, Keyframe

# ColorManager slot order used by GetLaneColorIndex: index 1/2/3 =
# Primary/Secondary/Middle (0 = the transparent/white "no slot" sentinel,
# never produced for keycounts 1..10).
SLOT_PRIMARY = 1
SLOT_SECONDARY = 2
SLOT_MIDDLE = 3
_SLOT_NAME = {SLOT_PRIMARY: 'primary', SLOT_SECONDARY: 'secondary',
              SLOT_MIDDLE: 'middle'}

# fluXis Theme accent defaults (#RRGGBB). ColorManager seeds each slot
# from Theme.GetLaneColor(1/2/3).Lighten(.2f); GetLaneColor derives
# Middle from Primary.Lighten(.4f). Precomputed here so we don't carry a
# color-space helper just to reproduce two constants.
_THEME_PRIMARY = (0x6F, 0x6F, 0xE2)
_THEME_SECONDARY = (0xAF, 0x59, 0xCF)


def _lighten(rgb, amount):
    """osu.Framework Colour4.Lighten: scale toward white by `amount`
    (raw linear approximation on 0-255 channels -- fluXis lightens in
    linear space, but these are seed defaults only, overridden the moment
    a map supplies `colors`)."""
    return tuple(min(255, round(c * (1 + amount) + 255 * amount)) for c in rgb)


def _default_slot_colors():
    primary = _lighten(_THEME_PRIMARY, 0.2)
    secondary = _lighten(_THEME_SECONDARY, 0.2)
    middle = _lighten(_lighten(_THEME_PRIMARY, 0.4), 0.2)
    return {SLOT_PRIMARY: primary, SLOT_SECONDARY: secondary,
            SLOT_MIDDLE: middle}


def lane_color_index(lane, keycount) -> int:
    """Port Theme.GetLaneColorIndex: 1-based `lane` -> slot (1/2/3) for a
    given keymode. Symmetric patterns keep mirrored lanes on the same
    accent. Keycounts outside 1..10 fall back to the centre (Middle)."""
    match keycount:
        case 1:
            return SLOT_MIDDLE
        case 2:
            return SLOT_SECONDARY
        case 3:
            return SLOT_SECONDARY if lane in (1, 3) else SLOT_MIDDLE
        case 4:
            return SLOT_PRIMARY if lane in (1, 4) else SLOT_SECONDARY
        case 5:
            if lane in (1, 5):
                return SLOT_PRIMARY
            if lane in (2, 4):
                return SLOT_SECONDARY
            return SLOT_MIDDLE
        case 6:
            return SLOT_SECONDARY if lane in (1, 3, 4, 6) else SLOT_PRIMARY
        case 7:
            if lane in (2, 6):
                return SLOT_PRIMARY
            if lane in (1, 3, 5, 7):
                return SLOT_SECONDARY
            return SLOT_MIDDLE
        case 8:
            if lane in (2, 7):
                return SLOT_PRIMARY
            if lane in (1, 3, 6, 8):
                return SLOT_SECONDARY
            return SLOT_MIDDLE
        case 9:
            if lane in (1, 3, 7, 9):
                return SLOT_PRIMARY
            if lane in (2, 4, 6, 8):
                return SLOT_SECONDARY
            return SLOT_MIDDLE
        case 10:
            if lane in (1, 3, 8, 10):
                return SLOT_PRIMARY
            if lane in (2, 4, 7, 9):
                return SLOT_SECONDARY
            return SLOT_MIDDLE
        case _:
            return SLOT_MIDDLE


def _seed_slots(chart_colors):
    """Live slot colors after ColorManager seeding: map `colors` override
    where present (and non-transparent), theme default otherwise."""
    override = chart_colors or {}
    slots = _default_slot_colors()
    for slot, name in _SLOT_NAME.items():
        rgb = override.get(name)
        if rgb is not None:
            slots[slot] = rgb
    return slots


# Main-playfield colorfade only for now: extra playfields / subfields
# route by these indices, unmodeled until the multi-playfield renderer.
_MAIN_PLAYFIELD = 0
_MAIN_SUBFIELD = 0

# Which fade-flag / color-key pairs a colorfade event carries per slot.
_FADE_SPEC = {
    SLOT_PRIMARY:   ('fade-primary', 'primary'),
    SLOT_SECONDARY: ('fade-secondary', 'secondary'),
    SLOT_MIDDLE:    ('fade-middle', 'middle'),
}


def _rgba_to_rgb(obj):
    """A colorfade RGBA object (`{R,G,B,A}`, 0-1 floats) -> `(r,g,b)`
    0-255. Alpha is ignored: notes are opaque, only their hue fades."""
    if not isinstance(obj, dict):
        return None
    return tuple(round(max(0.0, min(1.0, float(obj.get(k, 1.0)))) * 255)
                 for k in ('R', 'G', 'B'))


def _slot_timeline(events, slot, seed_rgb):
    """One EventTimeline per accent slot from the main-playfield colorfade
    stream. Each event whose `fade-<slot>` flag is set contributes a
    keyframe easing toward its target `(r,g,b)` over `duration`; the rest
    value is the seeded slot color, held before the first fade and
    between fades (ColorManager.TransformTo leaves the property at its
    last value)."""
    flag_key, color_key = _FADE_SPEC[slot]
    keyframes = []
    for e in events:
        if not e.get(flag_key):
            continue
        rgb = _rgba_to_rgb(e.get(color_key))
        if rgb is None:
            continue
        keyframes.append(Keyframe(
            t=float(e.get('time', 0.0)) / 1000.0,
            values=tuple(float(c) for c in rgb),
            duration=max(0.0, float(e.get('duration', 0.0))) / 1000.0,
            easing=int(e.get('ease', 0)),
        ))
    if not keyframes:
        return None
    return EventTimeline(keyframes, tuple(float(c) for c in seed_rgb))


class NotePalette:
    """Per-column note colors, sampleable at any time.

    `sample(t)` returns a length-`keycount` list of `(r,g,b)` tuples. When
    the chart has no colorfade events the palette is static and `sample`
    returns the same list every call (cheap identity path); otherwise the
    faded slots animate through their timelines.

    Reusable across games: any game whose notes are themed from a small
    palette of animated accents fits this shape (build the column->slot
    map and per-slot timelines however that game defines them).
    """

    def __init__(self, column_slots, seed_slots, slot_timelines):
        self._column_slots = tuple(column_slots)
        self._seed = dict(seed_slots)
        self._timelines = slot_timelines  # {slot: EventTimeline}
        self._animated = bool(slot_timelines)
        self._static = [self._seed[s] for s in self._column_slots]

    def __bool__(self):
        return True

    @property
    def animated(self) -> bool:
        return self._animated

    def static_colors(self) -> list:
        """The seed palette (t before any fade), one `(r,g,b)` per
        column. This is what `init` uses as `p.palette`."""
        return list(self._static)

    def sample(self, t_now: float) -> list:
        if not self._animated:
            return self._static
        slot_now = {}
        for slot, seed in self._seed.items():
            tl = self._timelines.get(slot)
            if tl is None:
                slot_now[slot] = seed
            else:
                r, g, b = tl.sample(t_now)
                slot_now[slot] = (round(r), round(g), round(b))
        return [slot_now[s] for s in self._column_slots]


def build_note_palette(chart_colors, keycount, colorfade_events=None):
    """Assemble the `NotePalette` for a fluXis chart.

    - `chart_colors`: the parsed `.fsc` `colors` dict ({slot: rgb|None}).
    - `keycount`: lane count (drives the lane->slot mapping).
    - `colorfade_events`: the raw `.ffx` `colorfade` stream, or None.
    """
    seed = _seed_slots(chart_colors)
    column_slots = [lane_color_index(col + 1, keycount)
                    for col in range(keycount)]

    main_events = [e for e in (colorfade_events or [])
                   if isinstance(e, dict)
                   and int(e.get('playfield', 0)) == _MAIN_PLAYFIELD
                   and int(e.get('subfield', 0)) == _MAIN_SUBFIELD]
    main_events.sort(key=lambda e: float(e.get('time', 0.0)))

    timelines = {}
    for slot in (SLOT_PRIMARY, SLOT_SECONDARY, SLOT_MIDDLE):
        tl = _slot_timeline(main_events, slot, seed[slot])
        if tl is not None:
            timelines[slot] = tl

    return NotePalette(column_slots, seed, timelines)
