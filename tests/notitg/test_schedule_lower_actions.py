"""Phase 2 specs: mod_actions lowering into preview lanes.

The oracle is the sim itself: firing the same handlers through a real
SimActor tick-by-tick must produce the same trajectories the lowering
computes at compile time. Unliftable handlers must go residue whole -
never partially lowered. Both engine command syntaxes are covered
(classic `verb,arg;...` chains and `%`-Lua bodies).
"""
import pytest

from analysis.games.notitg.schedule_lower import lower_actions
from analysis.games.notitg.sim.actor import SimActor


class _Host:
    def __init__(self, env=None):
        self.env = env or {}


class _StubEnv:
    def __init__(self, actions, message_commands, named_commands=None,
                 host_globals=None):
        self._staged_actions = actions
        self._message_commands = message_commands
        self._named_commands = named_commands or {}
        self._actors = {}
        self._host = _Host(host_globals)

    def named_actor_ids(self):
        return {}

    def actor(self, rec_id):
        self._actors[rec_id] = SimActor(now=0.0)
        return self._actors[rec_id]


CLASSIC_CHAIN = ('stoptweening;linear,0.5;x,100;sleep,0.25;'
                 'decelerate,1.0;x,-40;y,GAMESTATE:GetSongBeat()*2')
LUA_CHAIN = ('%function(self) self:stoptweening(); self:linear(0.5); '
             'self:x(100); self:sleep(0.25); self:decelerate(1.0); '
             'self:x(-40); self:y(GAMESTATE:GetSongBeat()*2) end')

CHAIN_POKES = [
    ('stoptweening', []), ('linear', [0.5]), ('x', [100.0]),
    ('sleep', [0.25]), ('decelerate', [1.0]), ('x', [-40.0]),
    ('y', [16.0]),
]


def _sim_oracle(body_pokes, fire_t, end_t):
    actor = SimActor(now=0.0)
    actor.update_to(fire_t)
    for verb, args in body_pokes:
        actor.poke(verb, args)
    t = fire_t
    while t < end_t:
        t = min(t + 1.0 / 60.0, end_t)
        actor.update_to(t)
    return {prop: lanes[0] for prop, lanes in actor._seg.items()}


@pytest.mark.parametrize('body', [CLASSIC_CHAIN, LUA_CHAIN],
                         ids=['classic', 'lua'])
def test_lowered_handler_matches_ticked_sim(body):
    env = _StubEnv([(2.0, 8.0, 'Go')], {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env)

    assert out.lifted_handlers == 1 and out.residue_handlers == 0
    oracle = _sim_oracle(CHAIN_POKES, fire_t=2.0, end_t=5.0)
    lanes = out.lanes[1]
    for prop in ('x', 'y'):
        for i in range(90):
            t = i * 0.05
            assert lanes[prop][0].sample(t) == pytest.approx(
                oracle[prop].sample(t), abs=1e-9), (prop, t)


def test_tail_append_queues_second_action_behind_first():
    env = _StubEnv(
        [(1.0, 4.0, 'A'), (1.2, 4.8, 'B')],
        {'A': [(1, 'linear,1.0;x,10')],
         'B': [(1, 'linear,1.0;x,20')]})
    env.actor(1)
    out = lower_actions(env)
    lane = out.lanes[1]['x'][0]
    assert lane.sample(2.0) == pytest.approx(10.0)
    assert lane.sample(2.5) == pytest.approx(15.0)
    assert lane.sample(3.0) == pytest.approx(20.0)


def test_stoptweening_resets_the_queue():
    env = _StubEnv(
        [(1.0, 4.0, 'A'), (1.5, 6.0, 'B')],
        {'A': [(1, 'linear,10.0;x,1000')],
         'B': [(1, 'stoptweening;linear,0.5;x,7')]})
    env.actor(1)
    out = lower_actions(env)
    lane = out.lanes[1]['x'][0]
    assert lane.sample(2.0) == pytest.approx(7.0)
    assert lane.sample(50.0) == pytest.approx(7.0)


def test_global_argument_resolves_from_post_load_env():
    env = _StubEnv(
        [(1.0, 4.0, 'Go')],
        {'Go': [(1, 'x,SCREEN_CENTER_X-220')]},
        host_globals={'SCREEN_CENTER_X': 320.0})
    env.actor(1)
    out = lower_actions(env)
    assert out.lanes[1]['x'][0].sample(1.0) == pytest.approx(100.0)


def test_unliftable_lua_statement_makes_whole_handler_residue():
    body = '%function(self) self:x(50); rig_helper(self) end'
    env = _StubEnv([(1.0, 4.0, 'Go')], {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env)
    assert out.residue_handlers == 1
    assert out.lanes == {}


def test_unknown_classic_verb_is_residue():
    env = _StubEnv([(1.0, 4.0, 'Go')],
                   {'Go': [(1, 'x,10;luaeffect,weird')]})
    env.actor(1)
    out = lower_actions(env)
    assert out.residue_handlers == 1
    assert out.lanes == {}


def test_closure_payloads_are_residue_actions():
    env = _StubEnv([(1.0, 4.0, lambda: None)], {})
    out = lower_actions(env)
    assert out.residue_actions == 1


def test_applymodifiers_becomes_applied_row():
    body = ("%function(self) self:ApplyModifiers('*-1 '"
            "..(GAMESTATE:GetSongBeat()*10)..' drunk') end")
    env = _StubEnv([(2.5, 10.0, 'Mods')], {'Mods': [(1, body)]})
    env.actor(1)
    out = lower_actions(env)
    assert out.applied == [(2.5, 10.0, '*-1 100 drunk', 1)]


def test_queuecommand_inlines_named_command():
    env = _StubEnv(
        [(1.0, 4.0, 'Go')],
        {'Go': [(1, 'linear,0.5;x,5;queuecommand,More')]},
        named_commands={1: {'More': 'linear,0.5;x,9'}})
    env.actor(1)
    out = lower_actions(env)
    lane = out.lanes[1]['x'][0]
    assert lane.sample(1.5) == pytest.approx(5.0)
    assert lane.sample(2.0) == pytest.approx(9.0)


def test_unknown_named_command_is_residue():
    env = _StubEnv([(1.0, 4.0, 'Go')],
                   {'Go': [(1, 'queuecommand,Nope')]})
    env.actor(1)
    out = lower_actions(env)
    assert out.residue_handlers == 1


def test_diffuse_lowers_color_lanes_and_alpha():
    env = _StubEnv([(1.0, 4.0, 'Go')],
                   {'Go': [(1, 'linear,1.0;diffuse,1,0.5,0,0.25')]})
    env.actor(1)
    out = lower_actions(env)
    color = out.lanes[1]['color']
    assert len(color) == 3
    assert [lane.sample(2.0) for lane in color] == \
        [pytest.approx(v) for v in (1.0, 0.5, 0.0)]
    assert out.lanes[1]['alpha'][0].sample(2.0) == pytest.approx(0.25)


def test_addx_is_relative_to_running_state():
    env = _StubEnv(
        [(1.0, 4.0, 'A'), (2.0, 8.0, 'B')],
        {'A': [(1, 'x,50')], 'B': [(1, 'addx,25')]})
    env.actor(1)
    out = lower_actions(env)
    lane = out.lanes[1]['x'][0]
    assert lane.sample(1.5) == pytest.approx(50.0)
    assert lane.sample(2.5) == pytest.approx(75.0)


def test_non_self_receiver_resolves_named_actor():
    class _NamedEnv(_StubEnv):
        def named_actor_ids(self):
            return {7: 'P1'}
    body = '%function(self) P1:linear(0.5); P1:x(60) end'
    env = _NamedEnv([(1.0, 4.0, 'Go')], {'Go': [(2, body)]})
    env.actor(2)
    env.actor(7)
    out = lower_actions(env)
    assert out.lanes[7]['x'][0].sample(2.0) == pytest.approx(60.0)


def test_queuemessage_rebroadcasts_at_queue_time():
    env = _StubEnv(
        [(1.0, 4.0, 'Go')],
        {'Go': [(1, 'sleep,0.5;queuemessage,Later')],
         'Later': [(2, 'x,42')]})
    env.actor(1)
    env.actor(2)
    out = lower_actions(env, to_beats=lambda t: t * 4.0)
    lane = out.lanes[2]['x'][0]
    assert lane.sample(1.49) == 0.0
    assert lane.sample(1.5) == pytest.approx(42.0)


def test_queuemessage_without_to_beats_is_residue():
    env = _StubEnv(
        [(1.0, 4.0, 'Go')],
        {'Go': [(1, 'queuemessage,Later')],
         'Later': [(2, 'x,42')]})
    env.actor(1)
    env.actor(2)
    out = lower_actions(env)
    assert out.residue_handlers == 1
    assert 2 not in out.lanes


class _NamedEnv(_StubEnv):
    NAMES: dict = {}

    def named_actor_ids(self):
        return dict(self.NAMES)


def _named_env(names, actions, message_commands, **kwargs):
    env = _NamedEnv(actions, message_commands, **kwargs)
    _NamedEnv.NAMES = names
    for rec_id in names:
        env.actor(rec_id)
    return env


def test_mod_message_closure_defers_at_its_beat():
    body = ('%function(self) mod_message(10, function() '
            'holder:x(8); holder:sleep(1.0); holder:linear(1.0); '
            'holder:x(2) end) end')
    env = _named_env({7: 'holder'}, [(1.0, 4.0, 'Go')],
                     {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env, to_seconds=lambda beat: beat * 0.5)
    lane = out.lanes[7]['x'][0]
    assert lane.sample(5.0) == pytest.approx(8.0)
    assert lane.sample(5.9) == pytest.approx(8.0)
    assert lane.sample(6.5) == pytest.approx(5.0)
    assert lane.sample(7.5) == pytest.approx(2.0)


def test_mod_message_string_broadcasts_at_its_beat():
    body = "%function(self) mod_message(10, 'Later') end"
    env = _StubEnv([(1.0, 4.0, 'Go')],
                   {'Go': [(1, body)], 'Later': [(2, 'x,42')]})
    env.actor(1)
    env.actor(2)
    out = lower_actions(env, to_seconds=lambda beat: beat * 0.5)
    lane = out.lanes[2]['x'][0]
    assert lane.sample(4.9) == 0.0
    assert lane.sample(5.0) == pytest.approx(42.0)


def test_mod_message_without_to_seconds_is_residue():
    body = "%function(self) mod_message(10, 'Later') end"
    env = _StubEnv([(1.0, 4.0, 'Go')],
                   {'Go': [(1, body)], 'Later': [(2, 'x,42')]})
    env.actor(1)
    env.actor(2)
    out = lower_actions(env)
    assert out.residue_handlers >= 1
    assert 2 not in out.lanes


def test_const_local_resolves_args_and_captures_into_closures():
    body = ('%function(self) local m = 0.5 '
            'self:linear(m*2); self:x(4); '
            'mod_message(10, function() self:x(m*100) end) end')
    env = _StubEnv([(1.0, 4.0, 'Go')], {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env, to_seconds=lambda beat: beat * 0.5)
    lane = out.lanes[1]['x'][0]
    assert lane.sample(1.5) == pytest.approx(2.0)
    assert lane.sample(2.0) == pytest.approx(4.0)
    assert lane.sample(5.0) == pytest.approx(50.0)


def test_const_condition_picks_the_live_branch():
    body = ('%function(self) if nvidia then self:x(1) '
            'else self:x(2) end end')
    for host, want in (({'nvidia': True}, 1.0), ({}, 2.0)):
        env = _StubEnv([(1.0, 4.0, 'Go')], {'Go': [(1, body)]},
                       host_globals=host)
        env.actor(1)
        out = lower_actions(env)
        assert out.lanes[1]['x'][0].sample(2.0) == pytest.approx(want)


def test_literal_for_unrolls_with_dynamic_global_receivers():
    body = ("%function(self) for i=1,2 do "
            "_G['pos'..i]:x(-160+(240*i)) end end")
    env = _named_env({4: 'pos1', 5: 'pos2'}, [(1.0, 4.0, 'Go')],
                     {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env)
    assert out.lanes[4]['x'][0].sample(2.0) == pytest.approx(80.0)
    assert out.lanes[5]['x'][0].sample(2.0) == pytest.approx(320.0)


def test_residue_handler_harvests_seeds_and_deferrals():
    body = ('%function(self) holder:x(5); rig_helper(self); '
            "mod_message(10, function() holder:y(9) end) end")
    env = _named_env({7: 'holder'}, [(1.0, 4.0, 'Go')],
                     {'Go': [(1, body)]})
    env.actor(1)
    out = lower_actions(env, to_seconds=lambda beat: beat * 0.5)
    assert out.residue_handlers == 1
    assert out.seed_pokes == {(7, 'x'): [(1.0, 5.0)]}
    assert out.lanes[7]['y'][0].sample(5.0) == pytest.approx(9.0)


def test_load_body_yields_deferrals_and_registrations_only():
    body = ('%function(self) self:x(999); table.insert(crew, self); '
            "mod_message(10, function() holder:x(3) end) end")
    env = _named_env({7: 'holder'}, [], {})
    env.actor(1)
    env._load_bodies = [(1, body)]
    env._load_seconds = 0.25
    out = lower_actions(env, to_seconds=lambda beat: beat * 0.5)
    assert out.registrations == {'crew': [(0.25, 1)]}
    assert 1 not in out.lanes            # executed state, not re-lowered
    assert out.lanes[7]['x'][0].sample(5.0) == pytest.approx(3.0)
