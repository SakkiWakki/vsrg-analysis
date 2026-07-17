"""Sprite-content-linter heuristics on synthetic sprites and sheets.

Each test hand-builds a `_Sprite` (real `EventTimeline`s plus a tiny PNG
sheet written to a tmp dir) whose logical/drawn size and source-frame
colour are computable by inspection, then asserts the finding it should
or should not produce. The point is to pin every heuristic's threshold
and its known guards - the authored-by-design zoomto exemption and the
transposed-grid detection especially - without compiling a chart.
"""
import numpy as np
import pytest
from PIL import Image

from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.storyboard.asset_size import AssetSizeSpec
from tools import sprite_lint as sl


# --- builders -------------------------------------------------------

def _timeline(pairs, rest=0.0):
    """An EventTimeline of instantaneous keyframes (a step at each t)."""
    keyframes = [Keyframe(t=t, values=(v,), duration=0.0, easing=0)
                 for t, v in pairs]
    return EventTimeline(keyframes, rest=(rest,))


def _color_timeline(pairs, rest=(1.0, 1.0, 1.0)):
    keyframes = [Keyframe(t=t, values=rgb, duration=0.0, easing=0)
                 for t, rgb in pairs]
    return EventTimeline(keyframes, rest=rest)


def _rests():
    """Rest timelines for a plain, always-visible, unscaled sprite."""
    return {
        'alpha': _timeline([], rest=1.0),
        'hidden': _timeline([], rest=0.0),
        'scale_x': _timeline([], rest=1.0),
        'scale_y': _timeline([], rest=1.0),
        'size_x': _timeline([], rest=sl._SIZE_UNSET - 1.0),
        'size_y': _timeline([], rest=sl._SIZE_UNSET - 1.0),
        'color': _color_timeline([]),
    }


def _sprite(asset, cols=1, rows=1, timelines=None, size_spec=None,
            state_pin=None, sheet_states=(), builtin=False):
    tl = _rests()
    tl.update(timelines or {})
    return sl._Sprite(
        name='synthetic', kind='sprite:test', asset=asset, builtin=builtin,
        sheet_cols=cols, sheet_rows=rows,
        size_spec=size_spec or AssetSizeSpec(cols=cols, rows=rows),
        state_pin=state_pin, sheet_states=sheet_states, timelines=tl)


def _write_png(path, px_w, px_h, painter):
    """A PNG at `path`; `painter(arr)` fills the (px_h, px_w, 3) array."""
    arr = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    painter(arr)
    Image.fromarray(arr, 'RGB').save(path)
    sl._PIXMAP_CACHE.pop(str(path), None)
    return str(path)


def _findings(sprite, chart_end=10.0):
    return sl.lint_sprite(sprite, chart_end)


def _by(findings, heuristic):
    return [f for f in findings if f.heuristic == heuristic]


@pytest.fixture(autouse=True)
def _clear_cache():
    sl._PIXMAP_CACHE.clear()
    yield
    sl._PIXMAP_CACHE.clear()


# --- ASSET ----------------------------------------------------------

def test_missing_asset_flagged_when_visible(tmp_path):
    sprite = _sprite(str(tmp_path / 'nope.png'))
    findings = _by(_findings(sprite), 'ASSET')
    assert len(findings) == 1
    assert findings[0].verdict == 'MISSING'
    assert 'does not exist' in findings[0].evidence['reason']


def test_directory_asset_reported_distinctly(tmp_path):
    (tmp_path / 'bg').mkdir()
    sprite = _sprite(str(tmp_path / 'bg'))
    findings = _by(_findings(sprite), 'ASSET')
    assert len(findings) == 1
    assert 'directory' in findings[0].evidence['reason']


def test_builtin_white_never_flagged():
    sprite = _sprite('white', builtin=True)
    assert _by(_findings(sprite), 'ASSET') == []


def test_missing_asset_not_flagged_when_never_visible(tmp_path):
    sprite = _sprite(str(tmp_path / 'nope.png'),
                     timelines={'alpha': _timeline([], rest=0.0)})
    assert _by(_findings(sprite), 'ASSET') == []


def test_good_asset_not_flagged(tmp_path):
    path = _write_png(tmp_path / 'ok.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    assert _by(_findings(_sprite(path)), 'ASSET') == []


# --- SHEET-GEOMETRY -------------------------------------------------

def test_uneven_grid_flagged(tmp_path):
    # 100px wide over a 3-col grid: 100 % 3 = 1, nearest boundary 1px off
    # is on the edge; 100 over 7 cols is 2px off -> flagged.
    path = _write_png(tmp_path / 'sheet 7x1.png', 100, 32,
                      lambda a: a.fill(120))
    sprite = _sprite(path, cols=7, rows=1,
                     size_spec=AssetSizeSpec(cols=7, rows=1))
    findings = _by(_findings(sprite), 'SHEET-GEOMETRY')
    assert any(f.verdict == 'UNEVEN-GRID' for f in findings)


def test_even_grid_not_flagged(tmp_path):
    path = _write_png(tmp_path / 'sheet 4x2.png', 128, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 128, 3))))
    sprite = _sprite(path, cols=4, rows=2,
                     size_spec=AssetSizeSpec(cols=4, rows=2))
    assert not any(f.verdict == 'UNEVEN-GRID'
                   for f in _by(_findings(sprite), 'SHEET-GEOMETRY'))


def test_frame_index_out_of_range_flagged(tmp_path):
    # 2x1 grid has frames 0,1; a pin at frame 5 over-runs.
    path = _write_png(tmp_path / 'two 2x1.png', 64, 32,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (32, 64, 3))))
    sprite = _sprite(path, cols=2, rows=1,
                     size_spec=AssetSizeSpec(cols=2, rows=1),
                     state_pin=_timeline([(0.0, 5.0)], rest=5.0))
    findings = _by(_findings(sprite), 'SHEET-GEOMETRY')
    over = [f for f in findings if f.verdict == 'FRAME-OUT-OF-RANGE']
    assert len(over) == 1
    assert 5 in over[0].evidence['out_of_range_frames']


def test_stale_manifest_logical_flagged(tmp_path):
    # File frame is 64x64; a manifest claiming 32x32 logical is stale.
    path = _write_png(tmp_path / 'm.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    sprite = _sprite(path, cols=1, rows=1,
                     size_spec=AssetSizeSpec(cols=1, rows=1,
                                             logical=(32.0, 32.0)))
    findings = _by(_findings(sprite), 'SHEET-GEOMETRY')
    assert any(f.verdict == 'STALE-MANIFEST' for f in findings)


# --- SOURCE-RECT (the screenshot signature) -------------------------

def _fuck_sheet(tmp_path, name, cols, rows):
    """A gat-style FUCK sheet: 128x96 noise, colourful per frame."""
    return _write_png(tmp_path / name, 128, 96,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (96, 128, 3))))


def test_sliver_stretched_by_zoomto_is_expected_by_design(tmp_path):
    # fuck 32x1: 4px-wide frame, zoomto(20, 480) -> the authored bar.
    path = _fuck_sheet(tmp_path, 'fuck 32x1.png', 32, 1)
    sprite = _sprite(
        path, cols=32, rows=1,
        size_spec=AssetSizeSpec(cols=32, rows=1),
        timelines={'size_x': _timeline([(0.0, 20.0)], rest=20.0),
                   'size_y': _timeline([(0.0, 480.0)], rest=480.0)})
    findings = _by(_findings(sprite), 'SOURCE-RECT')
    assert len(findings) == 1
    assert findings[0].verdict == 'EXPECTED-BY-DESIGN'
    assert sl._is_noise(findings[0])


def test_sliver_stretched_by_plain_scale_is_suspicious(tmp_path):
    # Same 4px frame blown to a 640x480 bar by SCALE, not zoomto -
    # no recorded absolute size explains it, so it is suspicious.
    path = _fuck_sheet(tmp_path, 'fuck 32x1.png', 32, 1)
    sprite = _sprite(
        path, cols=32, rows=1,
        size_spec=AssetSizeSpec(cols=32, rows=1),
        timelines={'scale_x': _timeline([(0.0, 160.0)], rest=160.0),
                   'scale_y': _timeline([(0.0, 160.0)], rest=160.0)})
    findings = _by(_findings(sprite), 'SOURCE-RECT')
    assert len(findings) == 1
    assert findings[0].verdict == 'SUSPICIOUS'
    assert not sl._is_noise(findings[0])


def test_normal_sprite_no_source_rect(tmp_path):
    path = _write_png(tmp_path / 'plain.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    assert _by(_findings(_sprite(path)), 'SOURCE-RECT') == []


def test_small_bar_below_bar_min_not_flagged(tmp_path):
    # A 4px frame drawn at only 30px is a tiny sliver, not a screen bar.
    path = _fuck_sheet(tmp_path, 'fuck 32x1.png', 32, 1)
    sprite = _sprite(
        path, cols=32, rows=1,
        size_spec=AssetSizeSpec(cols=32, rows=1),
        timelines={'size_x': _timeline([(0.0, 30.0)], rest=30.0),
                   'size_y': _timeline([(0.0, 30.0)], rest=30.0)})
    assert _by(_findings(sprite), 'SOURCE-RECT') == []


# --- ASPECT ---------------------------------------------------------

def test_aspect_equal_scale_no_size_is_never_distorted(tmp_path):
    # With equal scales and no absolute size, the drawn ratio equals the
    # logical ratio exactly, so a wide asset stays proportional. ASPECT
    # only fires when the drawn shape diverges from the logical one, which
    # requires an explaining cause (unequal scale / explicit size) that
    # the guards then exempt - so a well-formed element never trips it.
    path = _write_png(tmp_path / 'wide.png', 128, 16,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (16, 128, 3))))
    assert _by(_findings(_sprite(path, timelines={})), 'ASPECT') == []


def test_aspect_two_axis_setsize_squash_flagged(tmp_path):
    # A square 64x64 frame with a two-axis zoomto(400, 40) -> drawn ratio
    # 10 vs logical ratio 1 = 10x squash. Both sizes set to a distorting
    # ratio is the finding (a setsize computed from wrong source dims),
    # not an authored one-axis stretch.
    path = _write_png(tmp_path / 'sq.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    sprite = _sprite(path, timelines={
        'size_x': _timeline([(0.0, 400.0)], rest=400.0),
        'size_y': _timeline([(0.0, 40.0)], rest=40.0)})
    findings = _by(_findings(sprite), 'ASPECT')
    assert len(findings) == 1
    assert findings[0].verdict == 'DISTORTED'
    assert findings[0].evidence['factor'] >= 2.0


def test_aspect_unequal_scale_is_explained(tmp_path):
    path = _write_png(tmp_path / 'sq.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    # scale_x 8, scale_y 1 -> drawn 512x64 (ratio 8 vs logical 1) but the
    # independent scale poke EXPLAINS it: no ASPECT finding.
    sprite = _sprite(path, timelines={
        'scale_x': _timeline([(0.0, 8.0)], rest=8.0)})
    assert _by(_findings(sprite), 'ASPECT') == []


def test_aspect_explicit_size_on_one_axis_is_explained(tmp_path):
    path = _write_png(tmp_path / 'sq2.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    # size_x 512 (zoomtowidth) with natural height -> ratio 8, explained.
    sprite = _sprite(path, timelines={
        'size_x': _timeline([(0.0, 512.0)], rest=512.0)})
    assert _by(_findings(sprite), 'ASPECT') == []


# --- FLATNESS + transposed grid -------------------------------------

def test_flat_billboard_flagged(tmp_path):
    # A large near-uniform sprite where imagery belongs: a solid grey
    # 200x200 asset drawn at 300x300 (>10% of the 640x480 design area).
    path = _write_png(tmp_path / 'flat.png', 200, 200,
                      lambda a: a.fill(128))
    sprite = _sprite(path, timelines={
        'scale_x': _timeline([(0.0, 1.5)], rest=1.5),
        'scale_y': _timeline([(0.0, 1.5)], rest=1.5)})
    findings = _by(_findings(sprite), 'FLATNESS')
    assert len(findings) == 1
    assert findings[0].verdict == 'FLAT-BLOCK'


def test_flat_small_sprite_not_flagged(tmp_path):
    # Same flat asset drawn small (< 10% design area) is fine.
    path = _write_png(tmp_path / 'flat_small.png', 32, 32,
                      lambda a: a.fill(128))
    assert _by(_findings(_sprite(path)), 'FLATNESS') == []


def test_transposed_grid_reading_yields_imagery(tmp_path):
    # A 64x64 image of four solid VERTICAL colour strips. Read as a 4x1
    # grid, each frame is one 16x64 strip = a single solid colour (FLAT).
    # Read TRANSPOSED as 1x4, each frame is a 64x16 band crossing all
    # four strips = imagery. Recording the 4x1 grid is the mis-slice; the
    # transpose recovers imagery. Both frame axes have real extent (16px,
    # 64px) so this is a flat block, not a sliver.
    def strips(a):
        for i, val in enumerate((30, 90, 160, 230)):
            a[:, i * 16:(i + 1) * 16, :] = val
    path = _write_png(tmp_path / 'strips 4x1.png', 64, 64, strips)
    sprite = _sprite(path, cols=4, rows=1,
                     size_spec=AssetSizeSpec(cols=4, rows=1),
                     timelines={'scale_x': _timeline([(0.0, 6.0)], rest=6.0),
                                'scale_y': _timeline([(0.0, 6.0)], rest=6.0)})
    findings = _by(_findings(sprite), 'FLATNESS')
    assert len(findings) == 1
    assert findings[0].evidence['imagery_reading'] == 'transposed'
    assert findings[0].evidence['transposed_std'] > findings[0].evidence['recorded_std']


def test_textured_large_sprite_not_flat(tmp_path):
    path = _write_png(tmp_path / 'tex.png', 200, 200,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (200, 200, 3))))
    sprite = _sprite(path, timelines={
        'scale_x': _timeline([(0.0, 1.5)], rest=1.5),
        'scale_y': _timeline([(0.0, 1.5)], rest=1.5)})
    assert _by(_findings(sprite), 'FLATNESS') == []


# --- COLOR ----------------------------------------------------------

def test_near_black_tint_flagged(tmp_path):
    path = _write_png(tmp_path / 'c.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    sprite = _sprite(path, timelines={
        'color': _color_timeline([(0.0, (0.01, 0.01, 0.01))])})
    findings = _by(_findings(sprite), 'COLOR')
    assert len(findings) == 1
    assert findings[0].verdict == 'NEAR-BLACK'


def test_rest_white_color_never_flagged(tmp_path):
    path = _write_png(tmp_path / 'c2.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    # A rest-white colour (no keyframes) is not a tint bug.
    assert _by(_findings(_sprite(path)), 'COLOR') == []


def test_authored_bright_color_not_flagged(tmp_path):
    path = _write_png(tmp_path / 'c3.png', 64, 64,
                      lambda a: a.__setitem__(
                          (slice(None), slice(None)),
                          np.random.randint(0, 255, (64, 64, 3))))
    sprite = _sprite(path, timelines={
        'color': _color_timeline([(0.0, (0.5, 0.0, 0.0))])})
    # Red channel 0.5 is above the near-black floor -> not swallowed.
    assert _by(_findings(_sprite(path)), 'COLOR') == []


# --- ranking / noise separation -------------------------------------

def test_expected_by_design_ranks_below_genuine(tmp_path):
    design = _sprite(
        _fuck_sheet(tmp_path, 'fuck 32x1.png', 32, 1), cols=32, rows=1,
        size_spec=AssetSizeSpec(cols=32, rows=1),
        timelines={'size_x': _timeline([(0.0, 20.0)], rest=20.0),
                   'size_y': _timeline([(0.0, 480.0)], rest=480.0)})
    missing = _sprite(str(tmp_path / 'gone.png'))
    ranked = sl.lint_sprites([design, missing], 10.0)
    assert not sl._is_noise(ranked[0])
    assert sl._is_noise(ranked[-1])
