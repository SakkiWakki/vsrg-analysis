"""Base-field player selection: only the joined sides render base fields.

NotITG draws the joined players (at most P1/P2); players 3+ are chart-
created proxy SOURCES (`GetChild('PlayerP3')` + ApplyModifiers(pn)) whose
content reaches the screen strictly through ActorProxy copies. Emitting
them as base 'player' instances draws fields the engine never shows
(stacked at the center seat).
"""
from analysis.games.notitg.sim.producers import _base_players
from analysis.player.render.mods.channels import ModChannels


def _channels(players):
    return ModChannels({}, tuple(players))


def test_lone_player_stays_single():
    assert _base_players(_channels([0])) == [1]


def test_versus_pair():
    assert _base_players(_channels([0, 1])) == [1, 2]


def test_extra_proxy_source_players_are_not_base_fields():
    assert _base_players(_channels([0, 1, 2, 3, 4])) == [1, 2]


def test_player_two_only_mods_still_joins_both():
    assert _base_players(_channels([1])) == [1, 2]
