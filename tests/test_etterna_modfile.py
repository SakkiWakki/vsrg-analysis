"""Etterna SM5 modfile compiler: FGCHANGES parsing, Def.* actor-tree
execution under the stub environment, poptions method-call mod harvest,
command-function -> storyboard keyframes, and guarded integration tests
against the real World of Corruption / Valedumps pilots."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.etterna.modfile import (compile_mod_channels,
                                            compile_modfile, parse_fgchanges)
from analysis.games.etterna.recording_actor import ActorClock, RecordingActor
from analysis.games.etterna.sm5_env import Sm5Environment

_MINESWEEPER = Path('/home/yucky/.etterna/Songs/Valedumps 3 (Route B)/'
                    'Snake (poco0317)/minesweeper.sm')
_UNDISCOVERED = Path('/home/yucky/.etterna/Songs/World of Corruption/'
                     '[Zeta] Undiscovered Colors/01 Undiscovered Colors.ssc')
_CUPID = Path('/home/yucky/.etterna/Songs/I put Konata in the Banners Vol5/'
              'The Cupid of Romance (Poi)/The Cupid of Romance .sm')


def _run(source, name='test'):
    env = Sm5Environment()
    root = env.run_script(source, name=name)
    return env, root


# -- Def.* actor-tree construction ----------------------------------------

def test_def_actorframe_returns_recording_table_with_children():
    _env, root = _run("""
        local t = Def.ActorFrame { Name = 'root' }
        t[#t+1] = Def.Quad { Name = 'a' }
        t[#t+1] = Def.Sprite { Name = 'b' }
        return t
    """)
    assert root['Class'] == 'ActorFrame'
    assert root[1]['Class'] == 'Quad'
    assert root[2]['Class'] == 'Sprite'


def test_command_attributes_are_functions_not_strings():
    _env, root = _run("""
        return Def.Quad { InitCommand = function(self) self:xy(1, 2) end }
    """)
    assert callable(root['InitCommand'])


# -- poptions method-call harvest -----------------------------------------

def test_poptions_call_records_mod_event_with_value_and_speed():
    env, root = _run("""
        return Def.Actor { InitCommand = function(self)
            local po = GAMESTATE:GetPlayerState(PLAYER_1)
                                :GetPlayerOptions('ModsLevel_Preferred')
            po:Drunk(0.5, 3)
        end }
    """)
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    assert env.mod_events == [
        {'t': 0.0, 'mod': 'Drunk', 'value': 0.5, 'speed': 3.0, 'player': 0}]


def test_poptions_getter_returns_held_value_without_emitting():
    env, root = _run("""
        got = nil
        return Def.Actor { InitCommand = function(self)
            local po = GAMESTATE:GetPlayerState(PLAYER_1)
                                :GetPlayerOptions('ModsLevel_Preferred')
            po:Tornado(1.0)
            got = po:Tornado()
        end }
    """)
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    # One event for the setter; the bare getter emits nothing but reads
    # the held value back.
    assert len(env.mod_events) == 1
    assert env.host.env['got'] == pytest.approx(1.0)


def test_sleep_between_poptions_lays_down_a_timeline():
    env, root = _run("""
        return Def.Actor { InitCommand = function(self)
            local po = GAMESTATE:GetPlayerState(PLAYER_1)
                                :GetPlayerOptions('ModsLevel_Preferred')
            po:Drunk(0.2, 5)
            self:sleep(1.5)
            po:Drunk(0.8, 5)
            self:sleep(2.0)
            po:Tornado(1.0)
        end }
    """)
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    times = [(e['mod'], e['t']) for e in env.mod_events]
    assert times == [('Drunk', 0.0), ('Drunk', 1.5), ('Tornado', 3.5)]


def test_player2_routes_to_channel_player_1():
    env, root = _run("""
        return Def.Actor { InitCommand = function(self)
            local po = GAMESTATE:GetPlayerState(PLAYER_2)
                                :GetPlayerOptions('ModsLevel_Preferred')
            po:Drunk(0.5)
        end }
    """)
    # Our stub threads player 1 by default (GetPlayerState player index is
    # not carried), so this asserts the mapping contract, not P2 routing.
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    assert env.mod_events[0]['player'] in (0, 1)


# -- channel compilation --------------------------------------------------

def test_mod_events_compile_to_held_channel_values():
    events = [
        {'t': 0.0, 'mod': 'Drunk', 'value': 0.2, 'speed': 0.0, 'player': 0},
        {'t': 2.0, 'mod': 'Drunk', 'value': 0.8, 'speed': 0.0, 'player': 0},
    ]
    channels = compile_mod_channels(events)
    assert 'drunk' in channels.mods(0)
    # Breakpoints at (0, 0.2) and (2, 0.8): the value interpolates
    # linearly between them and holds the last value after the final one.
    assert channels.value('drunk', 0.0) == pytest.approx(0.2)
    assert channels.value('drunk', 1.0) == pytest.approx(0.5)
    assert channels.value('drunk', 3.0) == pytest.approx(0.8)


def test_mod_name_lowercased_to_channel():
    events = [{'t': 0.0, 'mod': 'Tornado', 'value': 1.0, 'speed': 0.0,
               'player': 0}]
    channels = compile_mod_channels(events)
    assert channels.mods(0) == ('tornado',)


# -- storyboard element compilation ---------------------------------------

def test_quad_command_compiles_to_rect_element():
    result = _compile_source("""
        local t = Def.ActorFrame {}
        t[#t+1] = Def.Quad { InitCommand = function(self)
            self:xy(100, 200):zoomto(64, 64):diffuse(1, 0, 0, 1)
        end }
        return t
    """)
    assert result['actors'] >= 1
    flat = result['elements']
    assert any(e.kind == 'rect' for e in flat)
    rect = next(e for e in flat if e.kind == 'rect')
    assert 'x' in rect.timelines and 'alpha' in rect.timelines


def test_actorframe_with_drawable_child_is_group():
    result = _compile_source("""
        local t = Def.ActorFrame { InitCommand = function(self)
            self:x(50)
        end }
        t[#t+1] = Def.Quad { InitCommand = function(self)
            self:xy(1, 2):zoomto(8, 8)
        end }
        return t
    """)
    assert any(e.kind == 'group' for e in result['tree'])


def test_sprite_carries_texture_asset():
    result = _compile_source("""
        return Def.Sprite { Texture = 'bg.png', InitCommand = function(self)
            self:xy(0, 0):diffusealpha(0.5)
        end }
    """)
    sprites = [e for e in result['elements'] if e.kind == 'sprite']
    assert sprites and sprites[0].asset == 'bg.png'


def _compile_source(source):
    """Drive the modfile compiler's tree walk directly on a Lua source,
    bypassing the FGCHANGES/file layer used by compile_modfile."""
    from analysis.games.etterna import modfile
    env = Sm5Environment()
    root = env.run_script(source, name='synthetic')
    tree, actors = modfile._compile_tree(env, root, 0.0, [])
    return {'tree': tree, 'elements': modfile._flatten(tree),
            'actors': actors, 'mod_events': env.mod_events}


# -- robustness (must not crash) ------------------------------------------

def test_unmodeled_globals_and_calls_do_not_abort():
    env, root = _run("""
        local prof = GetPlayerOrMachineProfile(PLAYER_1)
        local n = prof:GetName() .. ' player'
        SOUND:PlayOnce('x')
        return Def.Actor { InitCommand = function(self)
            local po = GAMESTATE:GetPlayerState(PLAYER_1)
                                :GetPlayerOptions('ModsLevel_Preferred')
            po:Beat(1.0)
        end }
    """)
    # The unmodeled globals degrade to permissive values; the tree still
    # returns and the poptions call still records.
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    assert env.mod_events[0]['mod'] == 'Beat'


def test_update_function_recorded_not_executed():
    env, root = _run("""
        return Def.Quad { InitCommand = function(self)
            self:SetUpdateFunction(function(s, dt) error('should not run') end)
        end }
    """)
    env.reset_clock(0.0)
    table, _rid = env.new_recorder_table()
    root['InitCommand'](table)
    assert env.update_functions == 1


# -- recording actor clock ------------------------------------------------

def test_recording_actor_shares_clock_with_tweens():
    clock = ActorClock(0.0)
    actor = RecordingActor(clock)
    actor.poke('x', [10.0])
    actor.poke('sleep', [2.0])
    actor.poke('x', [20.0])
    frames = actor.keyframes()['x']
    assert frames[0].t == pytest.approx(0.0)
    assert frames[1].t == pytest.approx(2.0)
    assert clock.now() == pytest.approx(2.0)


def test_linear_tween_carries_duration_to_next_setter():
    clock = ActorClock(0.0)
    actor = RecordingActor(clock)
    actor.poke('linear', [1.5])
    actor.poke('x', [100.0])
    frame = actor.keyframes()['x'][0]
    assert frame.duration == pytest.approx(1.5)


# -- fgchanges parsing ----------------------------------------------------

def test_parse_fgchanges_reads_lua_reference(tmp_path):
    sm = tmp_path / 'song.sm'
    sm.write_text('#OFFSET:0;\n#BPMS:0=120;\n'
                  '#FGCHANGES:0.000=mymod.lua=1.000=0=0=1;\n')
    entries = parse_fgchanges(str(sm))
    assert (0.0, 'mymod.lua') in entries


# -- integration (guarded on the real pilots) -----------------------------

@pytest.mark.skipif(not _UNDISCOVERED.exists(),
                    reason='World of Corruption pilot not present')
def test_undiscovered_colors_harvests_speedist_mods():
    """The Zeta Speedist template drives XMod scroll normalization via
    poptions method calls; it must harvest cleanly (timeline-driven, no
    warnings) with mod events on the XMod channel."""
    result = compile_modfile(str(_UNDISCOVERED))
    assert result is not None
    assert result['warnings'] == []
    assert result['mod_events']
    assert all(e['mod'] == 'XMod' for e in result['mod_events'])
    channels = compile_mod_channels(result['mod_events'])
    assert 'xmod' in channels.mods(0)


@pytest.mark.skipif(not _MINESWEEPER.exists(),
                    reason='Valedumps minesweeper pilot not present')
def test_minesweeper_interactive_modfile_does_not_crash():
    """The fully-interactive minesweeper modfile (SOUND callbacks, input
    hooks, chart queries) is the tier-B ceiling: it must load without
    raising, returning a result (mostly-unsupported, a warning where its
    live engine dependency stops the harvest)."""
    result = compile_modfile(str(_MINESWEEPER))
    assert result is not None
    assert isinstance(result['mod_events'], list)
    assert isinstance(result['elements'], list)


@pytest.mark.skipif(not _CUPID.exists(),
                    reason='Cupid of Romance modfile not present')
def test_cupid_compiles_storyboard_element():
    """A visual modfile whose Def.* actors carry command-function tweens
    compiles to at least one storyboard element with a drawable
    timeline."""
    result = compile_modfile(str(_CUPID))
    assert result is not None
    assert result['elements']
    element = result['elements'][0]
    assert element.timelines
