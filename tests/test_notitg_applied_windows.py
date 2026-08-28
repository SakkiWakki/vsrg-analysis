"""Applied-stream shaping (`sim.record`): clearall rows and window edges.

The classic template's per-frame reader applies ONE string per frame:
`clearall, <every live window re-applied>`. Within that frame the engine
runs clearall first (every seen mod retargets 0) and the trailing tokens
re-apply their mods, winning per-channel. The trailing tokens must
therefore never be dropped just because the row starts with clearall -
they carry the chart's persistent baseline (speed mods, always-on mods).
"""
import pytest

from analysis.games.notitg.sim.record import chase_events, coalesce_applied


def _rows(modstring, ticks, player=None, start=0.0, dt=1.0 / 60.0):
    return [(start + i * dt, (start + i * dt) * 2.0, modstring, player)
            for i in range(ticks)]


def test_clearall_row_keeps_trailing_mods():
    rows = _rows('clearall, *2 2x, *1000 50 drunk', ticks=120)
    windows = coalesce_applied(rows)
    drunk = [w for w in windows if w.name == 'drunk']
    assert drunk, 'mods after clearall in the same row must survive'
    assert {w.player for w in drunk} == {0, 1}
    for w in drunk:
        assert w.value == pytest.approx(0.5)
        assert w.speed == pytest.approx(1000.0)
        assert w.t_end - w.t_start == pytest.approx(119 / 60.0, abs=1e-6)


def test_clearall_row_keeps_trailing_speed_mods():
    rows = _rows('clearall, *2 2x', ticks=120)
    windows = coalesce_applied(rows)
    xmods = [w for w in windows if w.name == 'xmod']
    assert xmods, 'the baseline xmod rides the clearall row'
    for w in xmods:
        assert w.value == pytest.approx(2.0)


def test_clearall_still_reverts_mods_the_row_no_longer_carries():
    # 1s of drunk on the clearall row, then the row drops it: the
    # clearall must still retarget drunk to 0 at speed 1.
    rows = (_rows('clearall, *1000 50 drunk', ticks=60)
            + _rows('clearall', ticks=60, start=1.0))
    events = [e for e in chase_events(rows) if e.mod == 'drunk']
    assert events[0].value == pytest.approx(0.5)
    final = events[-1]
    assert final.value == 0.0
    assert final.speed == pytest.approx(1.0)
    assert final.beat == pytest.approx(1.0, abs=0.02)


def test_pure_clearall_row_shape_unchanged():
    rows = _rows('*1000 50 drunk', ticks=60) \
        + _rows('clearall', ticks=60, start=1.0)
    windows = coalesce_applied(rows)
    names = {w.name for w in windows}
    assert names == {'drunk'}


def test_large_magnitude_windows_survive_the_string_round_trip():
    # `%g` formats 1e6+ magnitudes in scientific notation
    # (`*1000000` -> `*1e+06`); the re-parse must accept it or the
    # window silently vanishes (the classic `*1000000 -9999999 cover`).
    rows = _rows('*1000000 -9999999 cover', ticks=60)
    windows = coalesce_applied(rows)
    assert windows
    from analysis.games.notitg.mod_channels import parse_modstring
    for w in windows:
        assert w.value == pytest.approx(-99999.99)
        (reparsed,) = parse_modstring(w.modstring)
        assert reparsed[0] == pytest.approx(-99999.99)
        assert reparsed[1] == pytest.approx(1000000.0)
        assert reparsed[2] == 'cover'


def test_clearall_reverts_the_scale_family_to_full_size():
    # `clearall` runs PlayerOptions::Init, and the scale family's default
    # there is 100%, not 0. Reverting zoomx to 0 collapses the field to a
    # point and HOLDS it: gat 2 stopped driving zoomx at 1:26 and its
    # notes stayed gone for the next 31 seconds.
    rows = (_rows('clearall, *10000 40 zoomx, *10000 50 drunk', ticks=60)
            + _rows('clearall', ticks=60, start=1.0))
    zoomx = [e for e in chase_events(rows) if e.mod == 'zoomx']
    drunk = [e for e in chase_events(rows) if e.mod == 'drunk']
    assert zoomx[0].value == pytest.approx(0.4)
    assert zoomx[-1].value == pytest.approx(1.0), 'zoomx clears to 100%'
    assert drunk[-1].value == 0.0, 'everything else still clears to 0'


def test_player_less_row_reaches_every_player_the_chart_mods():
    # `ApplyModifiers` with no `pn` is `ApplyToAllPlayers` -
    # FOREACH_PlayerNumber, and the fork's slot count runs past two. A
    # chart that mods P4 must get the player-less window on P4 too:
    # gat 2's revolt drove `invert`/`beat` player-less over four
    # playfields, and pinning the fan-out at (0, 1) left P3/P4 without
    # mods their siblings had.
    rows = (_rows('*1000 50 drunk', ticks=60)
            + _rows('*1000 20 dizzy', ticks=60, player=4))
    drunk = [w for w in coalesce_applied(rows) if w.name == 'drunk']
    assert {w.player for w in drunk} == {0, 1, 3}


def test_player_less_row_stays_on_both_sides_when_no_player_is_named():
    rows = _rows('*1000 50 drunk', ticks=60)
    drunk = [w for w in coalesce_applied(rows) if w.name == 'drunk']
    assert {w.player for w in drunk} == {0, 1}, 'two sides always exist'


def test_named_row_stays_on_its_own_channel():
    rows = (_rows('*1000 50 drunk', ticks=60, player=3)
            + _rows('*1000 20 dizzy', ticks=60, player=5))
    drunk = [w for w in coalesce_applied(rows) if w.name == 'drunk']
    assert {w.player for w in drunk} == {2}, 'a named row fans out to nobody'


def test_per_player_clearall_still_meets_the_all_players_window():
    # Why the expansion lives at ingestion: a player-less window and a
    # per-player clearall have to land on ONE key inside a frame so the
    # last call wins, as in the engine. P4 clears; P3 keeps its value.
    rows = (_rows('*1000 50 drunk', ticks=60)
            + _rows('*1000 20 dizzy', ticks=60, player=3)
            + _rows('clearall', ticks=60, player=4, start=1.0))
    drunk = {e.player: e.value for e in chase_events(rows) if e.mod == 'drunk'}
    assert drunk[3] == 0.0, 'P4 cleared the window it received player-less'
    assert drunk[2] == pytest.approx(0.5), 'P3 never cleared'
