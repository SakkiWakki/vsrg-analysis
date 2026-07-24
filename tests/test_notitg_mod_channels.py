"""Window -> channel resolution in notitg's `compile_mod_channels`.

gat-style templates run a per-frame `clearall` + re-apply reader: at any
instant a mod's target is the last-applied window covering it, else 0 at
`clearall` speed 1.0 (PlayerOptions::Init resets every m_SpeedfFoo to 1.0
and every value to 0). These tests fix the receptor-float semantics: a
window's END reverts at clearall speed, NOT the window's own `*S`, so a
`*10000` burst eases back over ~1s instead of snapping. Expected values
are hand-computed from the constant-rate `fapproach` chase.
"""
import pytest

from analysis.games.notitg.mod_channels import compile_mod_channels


def _window(start, end, modstring, player=None):
    return {'t_start': start, 't_end': end, 'modstring': modstring,
            'player': player}


def test_isolated_window_reverts_at_clearall_speed_not_its_own():
    # A *10000 window snaps drunk to 1.0 by its start (arrival ~instant),
    # holds to t=5, then reverts at clearall speed 1.0: 1.0 -> 0 over 1s,
    # reaching 0 at t=6. NOT the window's *10000 (which would snap).
    mc = compile_mod_channels([_window(0.0, 5.0, '*10000 100 drunk')])
    assert mc.value('drunk', 5.0) == pytest.approx(1.0)
    assert mc.value('drunk', 5.5) == pytest.approx(0.5)
    assert mc.value('drunk', 6.0) == pytest.approx(0.0)
    assert mc.value('drunk', 7.0) == pytest.approx(0.0)


def test_window_chases_in_at_its_own_speed():
    # Approach speed on the way IN is the window's *S: 0 -> 1.0 at speed 2
    # arrives after 1.0/2 = 0.5s.
    mc = compile_mod_channels([_window(0.0, 10.0, '*2 100 drunk')])
    assert mc.value('drunk', 0.0) == pytest.approx(0.0)
    assert mc.value('drunk', 0.25) == pytest.approx(0.5)
    assert mc.value('drunk', 0.5) == pytest.approx(1.0)
    assert mc.value('drunk', 5.0) == pytest.approx(1.0)


def test_snap_window_is_instant_in_and_floats_out():
    # *-1 (speed <= 0) snaps to target at the window start; the revert at
    # the end still floats at clearall speed 1.0.
    mc = compile_mod_channels([_window(0.0, 3.0, '*-1 100 flip')])
    assert mc.value('flip', 0.0) == pytest.approx(1.0)
    assert mc.value('flip', 3.0) == pytest.approx(1.0)
    assert mc.value('flip', 3.5) == pytest.approx(0.5)
    assert mc.value('flip', 4.0) == pytest.approx(0.0)


def test_explicit_no_window_reverts_at_that_windows_speed():
    # When the template pairs a rise with an explicit `no` window, THAT
    # window's speed governs the fall (this is gat's pattern - reverts are
    # template-owned). *5 100 mini in [0,2], then *2 no mini at t=2 pulls
    # mini 1.0 -> 0 at speed 2 => 0 at t=2.5.
    mc = compile_mod_channels([_window(0.0, 2.0, '*5 100 mini'),
                               _window(2.0, 6.0, '*2 no mini')])
    assert mc.value('mini', 2.0) == pytest.approx(1.0)
    assert mc.value('mini', 2.25) == pytest.approx(0.5)
    assert mc.value('mini', 2.5) == pytest.approx(0.0)


def test_zero_length_window_is_a_noop():
    # A start == end window covers no interval (half-open [start, end)):
    # the engine's next frame would clearall the one-frame spike away, so
    # it never drives the channel.
    mc = compile_mod_channels([_window(4.0, 4.0, '*100000 1000 drunk')])
    assert mc.value('drunk', 4.0) == pytest.approx(0.0)
    assert mc.value('drunk', 4.001) == pytest.approx(0.0)
    assert 'drunk' not in mc.mods(0)


def test_overlapping_window_suppresses_the_earlier_ends_revert():
    # Window A [0,4] drunk 100 (*-1 snap). Window B [2,8] drunk 50 (*-1)
    # is applied later in table order, so from t=2 it wins. At A's end
    # (t=4) drunk must NOT dip to 0 -- B still holds it at 0.5.
    mc = compile_mod_channels([_window(0.0, 4.0, '*-1 100 drunk'),
                               _window(2.0, 8.0, '*-1 50 drunk')])
    assert mc.value('drunk', 1.0) == pytest.approx(1.0)
    assert mc.value('drunk', 2.0) == pytest.approx(0.5)
    assert mc.value('drunk', 4.0) == pytest.approx(0.5)
    assert mc.value('drunk', 6.0) == pytest.approx(0.5)
    assert mc.value('drunk', 8.5) == pytest.approx(0.0)


def test_later_table_entry_wins_at_equal_time():
    # Two windows starting at the same time on the same mod: the later row
    # (higher order) wins, matching the reader re-applying in table order.
    mc = compile_mod_channels([_window(0.0, 5.0, '*-1 100 drunk'),
                               _window(0.0, 5.0, '*-1 30 drunk')])
    assert mc.value('drunk', 1.0) == pytest.approx(0.3)


def test_players_split_into_separate_channels():
    mc = compile_mod_channels([_window(0.0, 5.0, '*-1 100 drunk', player=1),
                               _window(0.0, 5.0, '*-1 40 drunk', player=2)])
    assert mc.value('drunk', 1.0, player=0) == pytest.approx(1.0)
    assert mc.value('drunk', 1.0, player=1) == pytest.approx(0.4)


def test_reversed_span_window_dropped():
    mc = compile_mod_channels([_window(5.0, 2.0, '*-1 100 drunk')])
    assert mc.mods(0) == ()


def test_pipe_args_keep_the_base_mod():
    # NotITG color-carrying mods format as `name|r|g|b` (exe format
    # strings: `stealthglow%d|%f|%f|%f`). The token must still drive the
    # base channel: prefix rules unchanged (bare name = 100%).
    from analysis.games.notitg.mod_channels import parse_modstring

    parsed = parse_modstring('*10000 stealthglow0|0.25|0|1')
    by_name = {name: (value, speed) for value, speed, name in parsed}
    assert by_name['stealthglow0'] == (1.0, 10000.0)


def test_pipe_args_become_rgb_companion_channels():
    from analysis.games.notitg.mod_channels import parse_modstring

    parsed = parse_modstring('*-1 stealthglow|0.25|0.5|1')
    by_name = {name: (value, speed) for value, speed, name in parsed}
    assert by_name['stealthglowred'][0] == pytest.approx(0.25)
    assert by_name['stealthglowgreen'][0] == pytest.approx(0.5)
    assert by_name['stealthglowblue'][0] == pytest.approx(1.0)
    # Companions ride the token's approach prefix.
    assert {by_name[n][1] for n in ('stealthglowred', 'stealthglowgreen',
                                    'stealthglowblue')} == {-1.0}


def test_pipe_args_on_numbered_column_variant():
    from analysis.games.notitg.mod_channels import parse_modstring

    parsed = parse_modstring('*10000 stealthglow2|0.9|0|1')
    names = {name for _v, _s, name in parsed}
    assert names == {'stealthglow2', 'stealthglow2red', 'stealthglow2green',
                     'stealthglow2blue'}


def test_pipe_token_inside_larger_modstring():
    from analysis.games.notitg.mod_channels import parse_modstring

    parsed = parse_modstring(
        '*10000 -20 stealth0,*10000 stealthglow0|0.9|0|1,*10000 50 tiny')
    by_name = {name: value for value, _s, name in parsed}
    assert by_name['stealth0'] == pytest.approx(-0.2)
    assert by_name['stealthglow0'] == pytest.approx(1.0)
    assert by_name['stealthglow0blue'] == pytest.approx(1.0)
    assert by_name['tiny'] == pytest.approx(0.5)
