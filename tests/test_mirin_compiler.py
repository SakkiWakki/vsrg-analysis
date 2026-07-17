"""Mirin declarative front-end compiler.

Detection, table harvest, and ease-baking exactness are checked against
the vendored template's OWN easing library and its `spec/*_spec.lua`
oracle: a chart authored with `ease`/`add`/`set`/`acc` is compiled and
its baked channels are sampled at spec timestamps, then compared to the
Lua ease functions evaluated directly (the same equality the template's
`ease_spec.lua` asserts) or to hand-computed constants.

Every fixture is a 60 BPM chart (offset 0) so beat == second and the
baked seconds-keyed channel samples at the ease progress directly.
"""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import mirin_compiler as mc
from analysis.games.notitg.mirin_compiler import (
    compile_mirin, is_mirin_chart)

_TEMPLATE = Path(__file__).resolve().parents[1] / (
    'refs/notitg/mirin-template')

pytestmark = pytest.mark.skipif(
    not (_TEMPLATE / 'template' / 'template.lua').is_file(),
    reason='vendored mirin-template clone absent (refs/ is gitignored)')

# 60 BPM, offset 0: one beat is one second, so a channel's second-keyed
# sample at t equals the ease evaluated at progress t/len.
_SM_HEADER = (
    '#TITLE:mirin fixture;\n'
    '#MUSIC:Song.ogg;\n'
    '#OFFSET:0.000;\n'
    '#BPMS:0.000=60.000;\n'
    '#STOPS:;\n'
    '#BGCHANGES:;\n'
    '#FGCHANGES:-10.000=template/main.xml=1.000=0=0=1=====,\n;\n'
    '#NOTES:\n     dance-single:\n     :\n     Beginner:\n     1:\n'
    '     0,0,0,0,0:\n'
    '0000\n1000\n0100\n0010\n;\n')


def _make_chart(tmp_path, mods_lua: str) -> Path:
    """A minimal Mirin chart directory: a 60 BPM Song.sm plus a
    lua/mods.lua carrying `mods_lua`. The compiler falls back to the
    vendored template dir (no bundled copy), so only the mods file
    varies between fixtures."""
    lua_dir = tmp_path / 'lua'
    lua_dir.mkdir()
    (lua_dir / 'mods.lua').write_text(mods_lua)
    sm_path = tmp_path / 'Song.sm'
    sm_path.write_text(_SM_HEADER)
    return sm_path


def _compile(tmp_path, mods_lua: str) -> dict:
    compiled = compile_mirin(str(_make_chart(tmp_path, mods_lua)))
    assert compiled is not None
    assert compiled['warnings'] == [], compiled['warnings']
    return compiled


def _ease(name: str):
    """One of the vendored template's Lua ease callables, run under a
    throwaway runtime - the exact oracle the compiler baked against."""
    rt = mc._new_runtime()
    g = rt.globals()
    rt.execute("math.pow = math.pow or function(a,b) return a^b end")
    rt.execute("xero = setmetatable({}, {__call=function() end})")
    rt.execute(
        "loadfile('%s/template/ease.lua')()"
        % str(_TEMPLATE).replace('\\', '/'))
    return g[name]


# --- detection -----------------------------------------------------------

def test_detects_bundled_template(tmp_path):
    (tmp_path / 'template').mkdir()
    (tmp_path / 'template' / 'template.lua').write_text(
        'setfenv(1, xero.strict)\n')
    assert is_mirin_chart(tmp_path) is True


def test_detects_template_beside_lua_dir(tmp_path):
    (tmp_path / 'template').mkdir()
    (tmp_path / 'template' / 'template.lua').write_text('local x = xero\n')
    (tmp_path / 'lua').mkdir()
    assert is_mirin_chart(tmp_path / 'lua') is True


def test_classic_chart_is_not_mirin(tmp_path):
    (tmp_path / 'default.xml').write_text(
        '<Actor Type="Quad" InitCommand="%mod_insert(0, 1, \'drunk\')"/>')
    assert is_mirin_chart(tmp_path) is False


def test_cat_framework_is_not_mirin(tmp_path):
    (tmp_path / 'default.xml').write_text(
        '<Actor InitCommand="%CatUpdater.add(function() end)"/>')
    assert is_mirin_chart(tmp_path) is False


# --- baseline / empty ----------------------------------------------------

def test_empty_mods_compiles_baseline(tmp_path):
    compiled = _compile(tmp_path, '-- no mods\n')
    assert compiled['dialect'] == 'mirin'
    assert compiled['mod_channels'].mods(0) == ()
    # The template's built-in setdefaults are always present.
    assert compiled['default_mods']['zoom'] == 100
    assert compiled['default_mods']['xmod'] == 1


def test_example_chart_compiles(tmp_path):
    """The vendored example chart (its own Song.sm + boilerplate
    mods.lua) compiles with no warnings and no channels."""
    compiled = compile_mirin(str(_TEMPLATE / 'Song.sm'))
    assert compiled is not None
    assert compiled['warnings'] == []
    assert compiled['mod_channels'].mods(0) == ()


# --- ease baking exactness (vs the Lua ease oracle) ----------------------

def test_outexpo_bakes_to_lua_curve(tmp_path):
    # ease_spec.lua: ease {0,1,outExpo,10000,'bumpy'} -> outExpo(.5)*10000
    # at half, sticks at 10000 after (outExpo(1) ~= 1 >= 0.5).
    compiled = _compile(tmp_path, "ease {0, 1, outExpo, 10000, 'bumpy'}\n")
    channels = compiled['mod_channels']
    out_expo = _ease('outExpo')
    assert channels.value('bumpy', 0.5, 0) == pytest.approx(
        out_expo(0.5) * 10000, rel=1e-3)
    assert channels.value('bumpy', 1.0, 0) == pytest.approx(10000, rel=1e-6)
    assert channels.value('bumpy', 2.0, 0) == pytest.approx(10000, rel=1e-6)


def test_baked_curve_tracks_ease_across_span(tmp_path):
    """A dense check: the baked channel matches the Lua ease at many
    interior progresses (the whole point of dense breakpoints)."""
    compiled = _compile(tmp_path, "ease {0, 2, inOutCubic, 500, 'drunk'}\n")
    channels = compiled['mod_channels']
    ease = _ease('inOutCubic')
    for progress in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9):
        t = progress * 2.0
        # rel loosens where the cubic is near-flat (tiny values amplify
        # relative error); abs bounds the true grid error (< 0.1 on 500).
        assert channels.value('drunk', t, 0) == pytest.approx(
            ease(progress) * 500, rel=2e-3, abs=0.1), progress


def test_transient_ease_returns_to_zero(tmp_path):
    # bounce(1) == 0 < 0.5 -> transient: peaks then returns to rest.
    compiled = _compile(tmp_path, "ease {0, 2, bounce, 100, 'drunk'}\n")
    channels = compiled['mod_channels']
    bounce = _ease('bounce')
    assert channels.value('drunk', 1.0, 0) == pytest.approx(
        bounce(0.5) * 100, rel=1e-3)  # == 100 at the peak
    assert channels.value('drunk', 2.0, 0) == pytest.approx(0.0, abs=1e-6)
    assert channels.value('drunk', 3.0, 0) == pytest.approx(0.0, abs=1e-6)


def test_additive_ease_stack(tmp_path):
    # ease_spec.lua additive-stack case, inBounce sticks (inBounce(1)==1).
    compiled = _compile(
        tmp_path,
        "ease {0, 1, inBounce, 10000, 'bumpy'}\n"
        "ease {0.5, 1, inBounce, 3000, 'bumpy'}\n")
    channels = compiled['mod_channels']
    in_bounce = _ease('inBounce')
    assert channels.value('bumpy', 0.5, 0) == pytest.approx(
        in_bounce(0.5) * 10000, rel=2e-3)
    assert channels.value('bumpy', 1.0, 0) == pytest.approx(
        10000 - in_bounce(0.5) * 7000, rel=2e-3)
    assert channels.value('bumpy', 1.5, 0) == pytest.approx(3000, rel=1e-6)


def test_most_recent_ease_wins(tmp_path):
    # ease_spec.lua: overlapping absolute eases resolve to the last one's
    # target (activation-delta math), here 800 at beat 10.
    mods = '\n'.join(
        "ease {0, %d, outExpo, %d, 'dizzy'}" % (length, length * 100)
        for length in (1, 4, 8, 6, 2, 9, 3, 5, 8))
    compiled = _compile(tmp_path, mods + '\n')
    assert compiled['mod_channels'].value('dizzy', 10.0, 0) == pytest.approx(
        800, rel=1e-6)


# --- set / acc -----------------------------------------------------------

def test_set_is_instant_absolute(tmp_path):
    compiled = _compile(tmp_path, "set {2, 50, 'drunk'}\n")
    channels = compiled['mod_channels']
    assert channels.value('drunk', 1.999, 0) == pytest.approx(0.0, abs=1e-6)
    assert channels.value('drunk', 2.0, 0) == pytest.approx(50, rel=1e-6)
    assert channels.value('drunk', 5.0, 0) == pytest.approx(50, rel=1e-6)


def test_acc_is_relative_set(tmp_path):
    compiled = _compile(
        tmp_path,
        "set {0, 50, 'drunk'}\nacc {1, 25, 'drunk'}\nset {2, 10, 'drunk'}\n")
    channels = compiled['mod_channels']
    assert channels.value('drunk', 0.5, 0) == pytest.approx(50, rel=1e-6)
    assert channels.value('drunk', 1.5, 0) == pytest.approx(75, rel=1e-6)
    assert channels.value('drunk', 2.5, 0) == pytest.approx(10, rel=1e-6)


def test_table_percent_jumps_then_eases(tmp_path):
    # ease {1,1,linear,{100,0},'bumpy'}: jump to 100 at start, ease to 0.
    compiled = _compile(
        tmp_path, "ease {1, 1, linear, {100, 0}, 'bumpy'}\n")
    channels = compiled['mod_channels']
    assert channels.value('bumpy', 1.0, 0) == pytest.approx(100, rel=1e-6)
    assert channels.value('bumpy', 1.5, 0) == pytest.approx(50, rel=1e-3)
    assert channels.value('bumpy', 2.0, 0) == pytest.approx(0.0, abs=1e-6)


# --- plr targeting -> per-player channels --------------------------------

def test_plr_splits_into_per_player_channels(tmp_path):
    compiled = _compile(
        tmp_path,
        "ease {0, 1, linear, 100, 'invert', plr = 1}\n"
        "ease {0, 1, linear, -100, 'invert', plr = 2}\n")
    channels = compiled['mod_channels']
    # Mirin players are 1-based; our channels are 0-based (pn - 1).
    assert channels.players == (0, 1)
    assert channels.value('invert', 1.0, 0) == pytest.approx(100, rel=1e-6)
    assert channels.value('invert', 1.0, 1) == pytest.approx(-100, rel=1e-6)
    assert channels.value('invert', 0.5, 0) == pytest.approx(50, rel=1e-3)


def test_default_plr_targets_both_players(tmp_path):
    compiled = _compile(tmp_path, "set {0, 40, 'drunk'}\n")
    channels = compiled['mod_channels']
    assert channels.players == (0, 1)
    assert channels.value('drunk', 1.0, 0) == pytest.approx(40, rel=1e-6)
    assert channels.value('drunk', 1.0, 1) == pytest.approx(40, rel=1e-6)


# --- add (relative) ------------------------------------------------------

def test_add_layers_relative_delta(tmp_path):
    # set drunk to 30, then add 20 with a sticking ease -> 50 committed.
    compiled = _compile(
        tmp_path,
        "set {0, 30, 'drunk'}\nadd {1, 1, outExpo, 20, 'drunk'}\n")
    channels = compiled['mod_channels']
    assert channels.value('drunk', 0.5, 0) == pytest.approx(30, rel=1e-6)
    assert channels.value('drunk', 2.0, 0) == pytest.approx(50, rel=1e-6)


# --- default_mods baseline -----------------------------------------------

def test_default_mods_seed_channel_rest(tmp_path):
    # zoom defaults to 100; a set to 200 rides on top of that baseline,
    # and the channel rests at 100 before the set.
    compiled = _compile(tmp_path, "aux('zoom')\nset {2, 200, 'zoom'}\n")
    channels = compiled['mod_channels']
    assert channels.value('zoom', 0.0, 0) == pytest.approx(100, rel=1e-6)
    assert channels.value('zoom', 2.5, 0) == pytest.approx(200, rel=1e-6)


# --- deferred tail -------------------------------------------------------

def test_perframe_and_func_go_to_deferred(tmp_path):
    compiled = _compile(
        tmp_path,
        "func {1, function() end}\n"
        "perframe {2, 3, function(beat, poptions) end}\n")
    kinds = sorted(d['kind'] for d in compiled['deferred'])
    assert kinds == ['func', 'perframe']
    perframe = next(d for d in compiled['deferred'] if d['kind'] == 'perframe')
    assert perframe['beat'] == pytest.approx(2.0)
    assert perframe['len_beats'] == pytest.approx(3.0)
    # A one-shot func has no length; declarative eases never appear here.
    func = next(d for d in compiled['deferred'] if d['kind'] == 'func')
    assert func['len_beats'] == 0.0


def test_declarative_mods_are_not_deferred(tmp_path):
    compiled = _compile(tmp_path, "ease {0, 1, outExpo, 100, 'drunk'}\n")
    assert compiled['deferred'] == []
    assert 'drunk' in compiled['mod_channels'].mods(0)


# --- node / definemod harvest --------------------------------------------

def test_chart_definemod_surfaced_as_node(tmp_path):
    compiled = _compile(
        tmp_path,
        "definemod {'wobble', function(p) return p, -p end,"
        " 'drunk', 'tornado'}\n")
    wobble = [n for n in compiled['nodes'] if n['inputs'] == ['wobble']]
    assert len(wobble) == 1
    assert wobble[0]['outputs'] == ['drunk', 'tornado']


def test_builtin_fanout_nodes_are_not_surfaced(tmp_path):
    # An empty chart still carries the template's built-in movex/zoom/xmod
    # fan-out nodes; none should appear as chart-authored nodes.
    compiled = _compile(tmp_path, '-- empty\n')
    assert compiled['nodes'] == []


# --- robustness ----------------------------------------------------------

def test_non_mirin_chart_returns_none(tmp_path):
    (tmp_path / 'lua').mkdir()
    (tmp_path / 'lua' / 'mods.lua').write_text("set {0, 1, 'drunk'}\n")
    (tmp_path / 'default.xml').write_text('<Actor/>')  # classic, not mirin
    sm_path = tmp_path / 'Song.sm'
    sm_path.write_text(_SM_HEADER.replace('template/main.xml', 'default.xml'))
    # No template dir and no xero fingerprint -> not a mirin chart.
    assert compile_mirin(str(sm_path)) is None


def test_broken_mods_lua_never_raises(tmp_path):
    compiled = compile_mirin(str(_make_chart(
        tmp_path, "this is not valid lua @@@\n")))
    assert compiled is not None
    assert compiled['dialect'] == 'mirin'
    assert compiled['warnings']  # captured, not raised


def test_snap_keyframes_do_not_double_ease(tmp_path):
    """The baked channel must NOT run the approach compiler: an instant
    `set` reads exactly its value one sample later, no chase ramp."""
    compiled = _compile(tmp_path, "set {0, 100, 'drunk'}\n")
    channels = compiled['mod_channels']
    assert channels.value('drunk', 0.0, 0) == pytest.approx(100, rel=1e-6)
    assert channels.value('drunk', 0.001, 0) == pytest.approx(100, rel=1e-6)
