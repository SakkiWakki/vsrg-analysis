"""fluXis note color theming: `.fsc` colors parse, the accent-trio
lane->slot mapping, static palettes, colorfade timelines, and the
per-frame quantized palette consumer."""
import pytest

from analysis.games.fluxis.fsc_chart import _parse_colors, _parse_hex_rgb
from analysis.games.fluxis.note_palette import (SLOT_MIDDLE, SLOT_PRIMARY,
                                                SLOT_SECONDARY,
                                                build_note_palette,
                                                lane_color_index)
from analysis.games.fluxis.palette_effect import PaletteFadeEffect


# ── colors parse ------------------------------------------------------

def test_parse_hex_rgb_forms_and_garbage():
    assert _parse_hex_rgb('#D70000') == (0xD7, 0x00, 0x00)
    assert _parse_hex_rgb('AF1010') == (0xAF, 0x10, 0x10)
    assert _parse_hex_rgb('#fff') is None      # wrong length
    assert _parse_hex_rgb('#GGGGGG') is None   # non-hex
    assert _parse_hex_rgb(None) is None


def test_parse_colors_maps_all_slots():
    # MARENOL's real colors object.
    colors = _parse_colors({'accent': '#CB000A', 'primary': '#D70000',
                            'secondary': '#AF1010', 'middle': '#D70000'})
    assert colors['primary'] == (0xD7, 0, 0)
    assert colors['secondary'] == (0xAF, 0x10, 0x10)
    assert colors['middle'] == (0xD7, 0, 0)
    assert colors['accent'] == (0xCB, 0, 0x0A)


def test_parse_colors_missing_slots_are_none():
    colors = _parse_colors({'primary': '#010203'})
    assert colors['primary'] == (1, 2, 3)
    assert colors['secondary'] is None
    assert colors['middle'] is None
    assert _parse_colors(None) == {s: None for s in
                                   ('accent', 'primary', 'secondary', 'middle')}


# ── lane -> slot mapping (ports Theme.GetLaneColorIndex) ---------------

def test_lane_color_index_4k_outer_primary_inner_secondary():
    assert [lane_color_index(l, 4) for l in (1, 2, 3, 4)] == \
        [SLOT_PRIMARY, SLOT_SECONDARY, SLOT_SECONDARY, SLOT_PRIMARY]


def test_lane_color_index_7k_pattern():
    got = [lane_color_index(l, 7) for l in range(1, 8)]
    # lanes 2,6 -> primary; 1,3,5,7 -> secondary; 4 -> middle (centre).
    assert got == [SLOT_SECONDARY, SLOT_PRIMARY, SLOT_SECONDARY, SLOT_MIDDLE,
                   SLOT_SECONDARY, SLOT_PRIMARY, SLOT_SECONDARY]


def test_lane_color_index_low_keymodes_and_symmetry():
    assert lane_color_index(1, 1) == SLOT_MIDDLE
    assert lane_color_index(1, 2) == SLOT_SECONDARY
    # 5k is left/right symmetric.
    left = [lane_color_index(l, 5) for l in (1, 2, 3)]
    right = [lane_color_index(l, 5) for l in (5, 4, 3)]
    assert left == right


def test_lane_color_index_out_of_range_falls_back_to_middle():
    assert lane_color_index(1, 16) == SLOT_MIDDLE


# ── static palette ----------------------------------------------------

def _colors(**over):
    base = {'accent': None, 'primary': None, 'secondary': None,
            'middle': None}
    base.update(over)
    return base


def test_static_palette_uses_chart_colors_per_column():
    chart_colors = _colors(primary=(10, 20, 30), secondary=(40, 50, 60),
                           middle=(70, 80, 90))
    pal = build_note_palette(chart_colors, 4, colorfade_events=None)
    cols = pal.static_colors()
    # 4k: outer=primary, inner=secondary.
    assert cols[0] == (10, 20, 30)
    assert cols[1] == (40, 50, 60)
    assert cols[2] == (40, 50, 60)
    assert cols[3] == (10, 20, 30)
    assert pal.animated is False
    # static path returns the same list object contents every call.
    assert pal.sample(0.0) == cols
    assert pal.sample(999.0) == cols


def test_missing_chart_colors_fall_back_to_theme_defaults():
    pal = build_note_palette(_colors(), 4, colorfade_events=None)
    cols = pal.static_colors()
    # All defaults are valid RGB and outer lanes share the primary default.
    assert cols[0] == cols[3]
    assert all(0 <= c <= 255 for rgb in cols for c in rgb)


# ── colorfade timelines -----------------------------------------------

def _fade(time_ms, r, g, b, duration=0.0, slot='primary'):
    ev = {'time': time_ms, 'duration': duration, 'ease': 0,
          'playfield': 0, 'subfield': 0,
          'fade-primary': False, 'fade-secondary': False,
          'fade-middle': False,
          'primary': {'R': 1.0, 'G': 1.0, 'B': 1.0, 'A': 1.0},
          'secondary': {'R': 1.0, 'G': 1.0, 'B': 1.0, 'A': 1.0},
          'middle': {'R': 1.0, 'G': 1.0, 'B': 1.0, 'A': 1.0}}
    ev[f'fade-{slot}'] = True
    ev[slot] = {'R': r, 'G': g, 'B': b, 'A': 1.0}
    return ev


def test_colorfade_animates_primary_columns_only():
    chart_colors = _colors(primary=(0, 0, 0), secondary=(40, 50, 60),
                           middle=(70, 80, 90))
    # Instant fade of primary to pure red at t=1000ms.
    events = [_fade(1000.0, 1.0, 0.0, 0.0, slot='primary')]
    pal = build_note_palette(chart_colors, 4, colorfade_events=events)
    assert pal.animated is True

    before = pal.sample(0.5)
    assert before[0] == (0, 0, 0)          # primary col, seed
    assert before[1] == (40, 50, 60)       # secondary col, untouched

    after = pal.sample(2.0)                 # 2000ms, past the fade
    assert after[0] == (255, 0, 0)          # primary col faded to red
    assert after[3] == (255, 0, 0)          # symmetric outer lane
    assert after[1] == (40, 50, 60)         # secondary col still static


def test_colorfade_eases_over_duration():
    chart_colors = _colors(primary=(0, 0, 0), secondary=(0, 0, 0),
                           middle=(0, 0, 0))
    # Ease black -> white over [1000, 2000] ms (ease 0 = linear).
    events = [_fade(1000.0, 1.0, 1.0, 1.0, duration=1000.0, slot='primary')]
    pal = build_note_palette(chart_colors, 4, colorfade_events=events)
    mid = pal.sample(1.5)[0]  # halfway
    assert all(abs(c - 127) <= 2 for c in mid)


def test_colorfade_ignores_non_main_playfield():
    chart_colors = _colors(primary=(0, 0, 0), secondary=(0, 0, 0),
                           middle=(0, 0, 0))
    ev = _fade(1000.0, 1.0, 0.0, 0.0, slot='primary')
    ev['playfield'] = 1  # a subfield / extra playfield -- unmodeled
    pal = build_note_palette(chart_colors, 4, colorfade_events=[ev])
    assert pal.animated is False
    assert pal.sample(2.0)[0] == (0, 0, 0)


# ── per-frame consumer (quantization) ---------------------------------

class _FakeCache:
    def __init__(self):
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1


class _FakeCtx:
    def __init__(self, palette_obj, t_now, cache):
        self.t_now = t_now
        self.sprite_cache = cache
        self.player = type('P', (), {'palette': None})()
        self._pal = palette_obj


def test_palette_effect_inactive_for_static_palette():
    pal = build_note_palette(_colors(), 4, colorfade_events=None)
    assert not PaletteFadeEffect(pal)          # falsy -> filtered out
    assert not PaletteFadeEffect(None)


def test_palette_effect_invalidates_only_on_quantized_change():
    chart_colors = _colors(primary=(0, 0, 0), secondary=(0, 0, 0),
                           middle=(0, 0, 0))
    # Slow linear fade black->white over 1s.
    events = [_fade(0.0, 1.0, 1.0, 1.0, duration=1000.0, slot='primary')]
    pal = build_note_palette(chart_colors, 4, colorfade_events=events)
    effect = PaletteFadeEffect(pal)
    assert effect

    cache = _FakeCache()
    ctx = _FakeCtx(pal, 0.0, cache)

    # Sample every ~1ms across the whole fade: far more frames than the
    # quantization can distinguish, so invalidations are bounded well
    # below the frame count.
    frames = 200
    for i in range(frames):
        ctx.t_now = i / (frames - 1)           # 0.0 .. 1.0s across the fade
        assert effect.at(ctx) is None          # draws nothing
        assert ctx.player.palette is not None

    # 32 levels -> at most ~32 distinct quantized palettes over the ramp,
    # nowhere near one-per-frame.
    assert cache.invalidations <= 40
    assert cache.invalidations < frames
    # Final palette reached white on the animated columns.
    assert ctx.player.palette[0] == (255, 255, 255)
