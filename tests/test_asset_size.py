"""Image-resolution regularization: the one funnel that turns raw pixel
size into an asset's LOGICAL frame size + grid, and StepMania's filename
conventions that feed it."""
import pytest

from analysis.games.notitg import sprite_sheet as ss
from analysis.player.render.storyboard.asset_size import (AssetSizeSpec,
                                                          LogicalSize, resolve)


# -- core precedence: manifest > grid > doubleres > res hint --------------

def test_plain_asset_logical_is_pixel_size():
    r = resolve(640, 480, AssetSizeSpec())
    assert r.natural == (640.0, 480.0)
    assert (r.cols, r.rows) == (1, 1)


def test_grid_divides_pixels_into_one_frame():
    # a 3x2 sheet at 384x512 -> one logical frame is 128x256.
    r = resolve(384, 512, AssetSizeSpec(cols=3, rows=2))
    assert r.natural == (128.0, 256.0)
    assert (r.cols, r.rows) == (3, 2)


def test_doubleres_halves_the_resolved_size():
    r = resolve(256, 256, AssetSizeSpec(doubleres=True))
    assert r.natural == (128.0, 128.0)


def test_doubleres_applies_after_the_grid():
    # grid gives 64x64 per frame, doubleres halves to 32x32.
    r = resolve(256, 256, AssetSizeSpec(cols=4, rows=4, doubleres=True))
    assert r.natural == (32.0, 32.0)


def test_res_hint_states_logical_size_directly():
    r = resolve(128, 128, AssetSizeSpec(res=(64, 64)))
    assert r.natural == (64.0, 64.0)


def test_res_hint_is_halved_by_doubleres():
    r = resolve(128, 128, AssetSizeSpec(res=(64, 64), doubleres=True))
    assert r.natural == (32.0, 32.0)


def test_grid_outranks_res_hint():
    # a real NxM sheet ignores a stray res hint (the grid fixed the size).
    r = resolve(256, 256, AssetSizeSpec(cols=2, rows=1, res=(999, 999)))
    assert r.natural == (128.0, 256.0)


def test_manifest_logical_outranks_everything():
    # an explicit manifest logical size wins over grid AND res hint; the
    # grid still describes how to crop frames out of the manifest size.
    r = resolve(256, 256, AssetSizeSpec(cols=2, rows=2, res=(9, 9),
                                        logical=(40, 30)))
    assert r.natural == (40.0, 30.0)
    assert (r.cols, r.rows) == (2, 2)


def test_manifest_logical_still_halved_by_doubleres():
    r = resolve(256, 256, AssetSizeSpec(logical=(80, 60), doubleres=True))
    assert r.natural == (40.0, 30.0)


def test_degenerate_grid_clamps_to_one():
    r = resolve(100, 100, AssetSizeSpec(cols=0, rows=-3))
    assert r.natural == (100.0, 100.0)
    assert (r.cols, r.rows) == (1, 1)


# -- notitg filename convention layer (gat's exact assets) ---------------

@pytest.mark.parametrize('name,px,logical', [
    # gat sheet frames == the standalone singles they mirror.
    ('shame_attack 3x2.png', (384, 512), (128, 256)),
    ('shame_idle 2x1.png', (256, 256), (128, 256)),
    # the degenerate FUCK sheet: 32 columns of 4px bars.
    ('fuck 32x1.png', (128, 96), (4, 96)),
    # same logical asset at two authoring resolutions, resolution-
    # independent: the actor code positions both identically.
    ('laugh 2x1.png', (600, 500), (300, 500)),
    ('--laugh 2x1.png', (512, 512), (256, 512)),
    # a plain single-frame image is its own logical size.
    ('bg.png', (640, 480), (640, 480)),
    ('casting.png', (240, 180), (240, 180)),
])
def test_notitg_filename_logical_size(name, px, logical):
    spec = ss.size_spec_from_filename(name)
    r = resolve(px[0], px[1], spec)
    assert r.natural == (float(logical[0]), float(logical[1]))


def test_notitg_doubleres_marker():
    spec = ss.size_spec_from_filename('char (doubleres).png')
    assert spec.doubleres is True
    assert resolve(256, 256, spec).natural == (128.0, 128.0)


def test_notitg_res_hint_marker():
    # a Government-Knows-era `(res WxH)` hint; the WxH must NOT read as a
    # stray grid token.
    spec = ss.size_spec_from_filename('sprite (res 64x48).png')
    assert spec.res == (64, 48)
    assert (spec.cols, spec.rows) == (1, 1)
    assert resolve(128, 96, spec).natural == (64.0, 48.0)


def test_notitg_manifest_logical_override():
    spec = ss.size_spec_from_filename('grid 2x2.png', logical=(10, 20))
    assert spec.logical == (10, 20)
    assert resolve(200, 200, spec).natural == (10.0, 20.0)


def test_logical_size_is_a_value_type():
    a = LogicalSize(4.0, 96.0, 32, 1)
    assert a == LogicalSize(4.0, 96.0, 32, 1)
    assert a.natural == (4.0, 96.0)
