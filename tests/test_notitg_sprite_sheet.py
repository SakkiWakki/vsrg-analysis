"""SM 3.95 sprite-sheet decode + sheet-aware storyboard rendering."""
from types import SimpleNamespace

import pytest

from analysis.games.notitg import sprite_sheet as sm_sheet
from analysis.player.render.storyboard import Storyboard, StoryboardEffect
from analysis.player.render.storyboard.model import Element, build_timelines
from analysis.player.render.storyboard.sprite_sheet import (frame_at_time,
                                                            frame_source_rect)


def _ctx(t=0.0, rect=(0, 0, 200, 200)):
    return SimpleNamespace(t_now=t, chart_rect=rect)


# ── filename grid decode -------------------------------------------------

@pytest.mark.parametrize('name,grid', [
    ('shame_idle 2x1.png', (2, 1)),
    ('--fuck 1x24.bmp', (1, 24)),
    ('fuck 32x1.png', (32, 1)),
    ('shame_attack 3x2.png', (3, 2)),
    ('bg.png', (1, 1)),
    ('casting.png', (1, 1)),
    ('/abs/path/laugh 2x1.png', (2, 1)),
])
def test_grid_from_filename(name, grid):
    assert sm_sheet.grid_from_filename(name) == grid


def test_grid_is_case_insensitive():
    assert sm_sheet.grid_from_filename('SHEET 4X3.PNG') == (4, 3)


# ── .sprite manifest state lists -----------------------------------------

def test_parse_sprite_states_oscillation():
    # shame idle.sprite: 0,1,0,1 with the first frame held longer.
    text = ('[Sprite]\nTexture=x 2x1.png\n'
            'Frame0000=0\nDelay0000=0.5\n'
            'Frame0001=1\nDelay0001=0.125\n'
            'Frame0002=0\nDelay0002=0.125\n'
            'Frame0003=1\nDelay0003=0.125\n')
    states = sm_sheet.parse_sprite_states(text, frame_count=2)
    assert states == ((0, 0.5), (1, 0.125), (0, 0.125), (1, 0.125))


def test_parse_sprite_states_stops_at_first_gap():
    # attack.sprite plays 0..4 then holds on 4 (delay 999).
    text = '\n'.join(f'Frame{i:04d}={i}\nDelay{i:04d}=0.04' for i in range(5))
    text += '\nFrame0005=2'  # no matching Delay0005 -> stops before it
    states = sm_sheet.parse_sprite_states(text, frame_count=6)
    assert [f for f, _d in states] == [0, 1, 2, 3, 4]


def test_parse_sprite_states_single_pin():
    # hurt.sprite pins frame 1 with an effectively-infinite delay.
    text = 'Frame0000=1\nDelay0000=999'
    assert sm_sheet.parse_sprite_states(text, frame_count=2) == ((1, 999.0),)


def test_parse_sprite_states_none_when_no_frames():
    assert sm_sheet.parse_sprite_states('[Sprite]\nTexture=x.png', 4) == ()


def test_default_states_sequential():
    assert sm_sheet.default_states(3) == ((0, 0.1), (1, 0.1), (2, 0.1))


# ── frame source rects (grid decode order: across a row, then down) ------

def test_frame_source_rect_row_major():
    # 3x2 sheet, 120x100 px -> 40x50 cells; index runs across then down.
    assert frame_source_rect(0, 120, 100, 3, 2) == (0, 0, 40, 50)
    assert frame_source_rect(2, 120, 100, 3, 2) == (80, 0, 40, 50)
    assert frame_source_rect(3, 120, 100, 3, 2) == (0, 50, 40, 50)
    assert frame_source_rect(5, 120, 100, 3, 2) == (80, 50, 40, 50)


def test_frame_source_rect_clamps_past_grid():
    assert frame_source_rect(99, 120, 100, 3, 2) == (80, 50, 40, 50)


# ── auto-animation over a state list -------------------------------------

def test_frame_at_time_steps_and_loops():
    states = ((0, 0.5), (1, 0.125), (0, 0.125), (1, 0.125))  # total 0.875
    assert frame_at_time(states, 0.0) == 0
    assert frame_at_time(states, 0.4) == 0
    assert frame_at_time(states, 0.5) == 1
    assert frame_at_time(states, 0.7) == 0
    assert frame_at_time(states, 0.8) == 1
    assert frame_at_time(states, 0.875) == 0   # wrapped back to start


def test_frame_at_time_single_state_holds():
    assert frame_at_time(((4, 999.0),), 100.0) == 4


def test_frame_at_time_empty_is_frame_zero():
    assert frame_at_time((), 5.0) == 0


# ── sheet-aware rendering ------------------------------------------------

def _sheet_png(tmp_path, cols, rows, cell=16):
    """A cols x rows sheet where every cell is a distinct solid colour, so
    a rendered crop's colour identifies which frame was drawn."""
    from PySide6.QtGui import QColor, QImage, QPainter
    img = QImage(cols * cell, rows * cell, QImage.Format.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    for row in range(rows):
        for col in range(cols):
            frame = col + row * cols
            p.fillRect(col * cell, row * cell, cell, cell,
                       QColor(frame * 20 % 256, 40, 200))
    p.end()
    path = tmp_path / f'sheet {cols}x{rows}.png'
    img.save(str(path))
    return str(path), cell


def _sheet_element(asset, cols, rows, cell, **overrides):
    fields = dict(
        kind='sprite', z=0, z_index=0, t_start=0.0, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.0, 0.0), timelines=build_timelines(),
        asset=asset, sheet_cols=cols, sheet_rows=rows)
    fields.update(overrides)
    return Element(**fields)


def _drawn_source_rect(tmp_path, element):
    from analysis.player.render.storyboard.render import StoryboardEffect

    class _Rec:
        def __init__(self):
            self.pixmap_calls = []

        def __getattr__(self, name):
            def rec(*args):
                if name == 'drawPixmap':
                    self.pixmap_calls.append(args)
            return rec

    sb = Storyboard(200, 200, 'height', (element,))
    eff = StoryboardEffect(sb)
    frame = eff.at(_ctx(t=0.0))
    rec = _Rec()
    for _z, draw in frame.draws:
        draw(_ctx(t=0.0), rec)
    return rec.pixmap_calls


def test_natural_size_is_one_frame_not_the_sheet(tmp_path):
    asset, cell = _sheet_png(tmp_path, 3, 2)
    el = _sheet_element(asset, 3, 2, cell,
                        sheet_states=sm_sheet.default_states(6))
    eff = StoryboardEffect(Storyboard(200, 200, 'height', (el,)))
    size = eff._element_size(el, 0.0)
    # sheet is 48x32; one frame is 16x16, NOT the whole sheet.
    assert size == (16.0, 16.0)


def test_source_rect_follows_auto_animation(tmp_path):
    asset, cell = _sheet_png(tmp_path, 3, 2)
    states = ((0, 0.1), (4, 0.1))  # frame 0 then frame 4
    el = _sheet_element(asset, 3, 2, cell, sheet_states=states)

    at_start = _drawn_source_rect(tmp_path, el)
    assert at_start, 'a sheet sprite should draw one crop'
    src = at_start[0][2]  # QRectF source arg of drawPixmap(dest, pm, src)
    assert (src.x(), src.y(), src.width(), src.height()) == (0, 0, 16, 16)


def test_state_pin_overrides_auto_animation(tmp_path):
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    asset, cell = _sheet_png(tmp_path, 3, 2)
    # auto-anim would show frame 0 at t=0; the pin forces frame 5.
    pin = EventTimeline([Keyframe(0.0, (5.0,), 0.0, 0)], rest=(0.0,))
    el = _sheet_element(asset, 3, 2, cell,
                        sheet_states=sm_sheet.default_states(6),
                        state_pin=pin)
    calls = _drawn_source_rect(tmp_path, el)
    src = calls[0][2]
    # frame 5 in a 3x2 sheet = col 2, row 1 -> (32, 16, 16, 16).
    assert (src.x(), src.y(), src.width(), src.height()) == (32, 16, 16, 16)


def test_plain_sprite_draws_whole_pixmap(tmp_path):
    asset, cell = _sheet_png(tmp_path, 1, 1)
    el = _sheet_element(asset, 1, 1, cell)  # 1x1 grid = not a sheet
    calls = _drawn_source_rect(tmp_path, el)
    src = calls[0][2]
    assert (src.width(), src.height()) == (16, 16)  # whole 1x1 pixmap


# ── resolution API in the draw path: logical size, not raw pixels --------

def test_doubleres_natural_size_is_halved(tmp_path):
    # a (doubleres) 64x64 texture logically occupies 32x32, not 64x64.
    from analysis.player.render.storyboard.asset_size import AssetSizeSpec
    asset, _cell = _sheet_png(tmp_path, 1, 1, cell=64)
    el = _sheet_element(asset, 1, 1, 64,
                        size_spec=AssetSizeSpec(doubleres=True))
    eff = StoryboardEffect(Storyboard(200, 200, 'height', (el,)))
    assert eff._element_size(el, 0.0) == (32.0, 32.0)


def test_zoomto_overrides_natural_with_absolute_size():
    # SM zoomto sets the on-screen size DIRECTLY: a 4px-wide frame with a
    # zoomto(20, 480) draws 20x480 regardless of its tiny logical size -
    # the gat FUCK-bar fullscreen mechanism.
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    from analysis.player.render.storyboard.render import _draw_size
    tl = {'size_x': EventTimeline([Keyframe(0.0, (20.0,), 0.0, 0)],
                                  rest=(-1.0,)),
          'size_y': EventTimeline([Keyframe(0.0, (480.0,), 0.0, 0)],
                                  rest=(-1.0,))}
    el = SimpleNamespace(timelines=tl, sample=lambda p, t: tl[p].sample(t))
    assert _draw_size(el, 0.0, (4.0, 96.0)) == (20.0, 480.0)


def test_unset_size_falls_back_to_natural():
    from analysis.player.render.effects.timeline import EventTimeline
    from analysis.player.render.storyboard.render import _draw_size
    tl = {'size_x': EventTimeline([], rest=(-1.0,)),
          'size_y': EventTimeline([], rest=(-1.0,))}
    el = SimpleNamespace(timelines=tl, sample=lambda p, t: tl[p].sample(t))
    assert _draw_size(el, 0.0, (128.0, 256.0)) == (128.0, 256.0)


def test_single_axis_zoomto_keeps_other_natural():
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe
    from analysis.player.render.storyboard.render import _draw_size
    tl = {'size_x': EventTimeline([], rest=(-1.0,)),
          'size_y': EventTimeline([Keyframe(0.0, (480.0,), 0.0, 0)],
                                  rest=(-1.0,))}
    el = SimpleNamespace(timelines=tl, sample=lambda p, t: tl[p].sample(t))
    assert _draw_size(el, 0.0, (100.0, 96.0)) == (100.0, 480.0)


# ── recorder: zoomto/setsize/zoomtowidth record absolute size ------------

def test_recorder_zoomto_records_both_axes():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=3.0)
    actor.poke('zoomto', [20, 480.0])
    kf = actor.keyframes()
    assert kf['size_x'][0].values == (20.0,)
    assert kf['size_y'][0].values == (480.0,)
    assert kf['size_x'][0].t == 3.0


def test_recorder_setsize_records_absolute_size():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=0.0)
    actor.poke('setsize', [640, 480])
    kf = actor.keyframes()
    assert (kf['size_x'][0].values, kf['size_y'][0].values) == ((640.0,),
                                                                (480.0,))


def test_recorder_zoomtowidth_sets_one_axis():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=0.0)
    actor.poke('zoomtowidth', [100])
    kf = actor.keyframes()
    assert kf['size_x'][0].values == (100.0,)
    assert 'size_y' not in kf


def test_recorder_zoomto_resolves_screen_constants():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=0.0)
    actor.poke('zoomto', [20, 'SCREEN_HEIGHT'])
    assert actor.keyframes()['size_y'][0].values == (480.0,)


# ── modfile compile: grid + .sprite states onto the element --------------

pytest.importorskip('lupa')


def _actor(xml, base_dir):
    from analysis.games.notitg import xml_actors
    actor = xml_actors.parse_actor_xml(xml).root
    actor._base_dir = base_dir
    return actor


def test_compile_sheet_sprite_from_nxm_filename(tmp_path):
    from analysis.games.notitg import modfile
    _sheet_png(tmp_path, 4, 3)  # writes 'sheet 4x3.png' beside base_dir
    actor = _actor('<Sprite Type="Sprite" Texture="sheet 4x3.png" '
                   'OnCommand="diffusealpha,1"/>', tmp_path)
    el = modfile._compile_actor(actor, 0.0, {}, None)
    assert el.kind == 'sprite'
    assert (el.sheet_cols, el.sheet_rows) == (4, 3)
    assert len(el.sheet_states) == 12          # default sequential anim
    assert el.state_pin is None


def test_compile_sprite_manifest_states(tmp_path):
    from analysis.games.notitg import modfile
    _sheet_png(tmp_path, 2, 1)  # 'sheet 2x1.png'
    (tmp_path / 'idle.sprite').write_text(
        '[Sprite]\nTexture=sheet 2x1.png\n'
        'Frame0000=0\nDelay0000=0.5\n'
        'Frame0001=1\nDelay0001=0.125\n'
        'Frame0002=0\nDelay0002=0.125\n')
    actor = _actor('<Sprite Type="Sprite" Texture="idle.sprite" '
                   'OnCommand="diffusealpha,1"/>', tmp_path)
    el = modfile._compile_actor(actor, 0.0, {}, None)
    # manifest states OVERRIDE the default sequence, and the asset is the
    # sheet the manifest points at (not the .sprite path).
    assert el.sheet_states == ((0, 0.5), (1, 0.125), (0, 0.125))
    assert el.asset.endswith('sheet 2x1.png')
    assert (el.sheet_cols, el.sheet_rows) == (2, 1)


def test_compile_plain_sprite_has_no_grid(tmp_path):
    from analysis.games.notitg import modfile
    from PySide6.QtGui import QImage
    img = QImage(8, 8, QImage.Format.Format_ARGB32)
    img.fill(0)
    img.save(str(tmp_path / 'plain.png'))
    actor = _actor('<Sprite Type="Sprite" Texture="plain.png" '
                   'OnCommand="diffusealpha,1"/>', tmp_path)
    el = modfile._compile_actor(actor, 0.0, {}, None)
    assert (el.sheet_cols, el.sheet_rows) == (1, 1)
    assert el.sheet_states == ()


# ── setstate / animate recording -> state pin ----------------------------

def test_recorder_setstate_pins_frame():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=2.0)
    actor.poke('setstate', [3])
    frames = actor.keyframes()['frame']
    assert len(frames) == 1
    assert frames[0].t == 2.0
    assert frames[0].values == (3.0,)


def test_recorder_animate_off_freezes_current_frame():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=0.0)
    actor.poke('setstate', [2])
    actor.poke('sleep', [1.0])
    actor.poke('animate', [0])            # freeze on frame 2 at t=1.0
    frames = actor.keyframes()['frame']
    assert [kf.values[0] for kf in frames] == [2.0, 2.0]
    assert frames[-1].t == 1.0


def test_recorder_animate_on_leaves_no_pin():
    from analysis.games.notitg.recording_actor import RecordingActor
    actor = RecordingActor(clock=0.0)
    actor.poke('animate', [1])
    assert 'frame' not in actor.keyframes()


def test_state_pin_from_recorded_setstate(tmp_path):
    from analysis.games.notitg import modfile, xml_actors
    from analysis.games.notitg.recording_actor import RecordingActor
    _sheet_png(tmp_path, 3, 2)
    actor = xml_actors.parse_actor_xml(
        '<Sprite Type="Sprite" Texture="sheet 3x2.png" '
        'InitCommand="%function(self) spr = self end"/>').root
    actor._base_dir = tmp_path
    # a closure poked setstate(4) at t=5 onto the bound global `spr`.
    recorder = RecordingActor(clock=5.0)
    recorder.poke('setstate', [4])
    named = {'spr': recorder.keyframes()}
    el = modfile._compile_actor(actor, 0.0, named, None)
    assert el.state_pin is not None
    assert el.state_pin.sample(5.0)[0] == pytest.approx(4.0)
