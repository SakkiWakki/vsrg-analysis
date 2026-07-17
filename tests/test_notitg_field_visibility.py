"""Per-copy capture scope + base-field visibility (scoping items 32/34).

gat has two field-copy SOURCES: ActorProxy copies of a player's NoteField
(notes + receptors only) and ActorFrameTexture screen captures (whole
screen, background included only in the chart's ShowAFTBG sections). The
compiler must (a) sample the real player's hidden state so the base field
is suppressed while proxies stand in, and (b) sample the AFT bg-in-capture
state so AFT copies carry the background only in the right sections.
These tests drive those compiler surfaces with synthetic XML - no gat
install needed."""
from __future__ import annotations

from analysis.games.notitg import modfile
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml


def _seconds_identity(beat):
    return float(beat)


def test_screen_child_player_records_hidden_pokes():
    """`SCREENMAN:GetTopScreen():GetChild('PlayerP1')` returns a recorder
    the chart can hide (`P1:hidden(1)`), and the poke is harvested."""
    env = StubEnvironment(start_beat=0.0)
    env.run("P1 = SCREENMAN:GetTopScreen():GetChild('PlayerP1')\n"
            "P1:hidden(1)", name='t')
    hidden = env.player_keyframes('PlayerP1').get('hidden')
    assert hidden and hidden[-1].values[0] == 1.0
    # A player never poked reports nothing.
    assert env.player_keyframes('PlayerP2') == {}


def test_base_field_hidden_timeline_tracks_show_hide():
    """A hide then a later show compile into a 1->0 base-hidden timeline."""
    root = parse_actor_xml(
        '<ActorFrame '
        'HideMessageCommand="%function(self) '
        'P1 = SCREENMAN:GetTopScreen():GetChild(\'PlayerP1\'); '
        'P1:hidden(1) end" '
        'ShowMessageCommand="%function(self) '
        'P1 = SCREENMAN:GetTopScreen():GetChild(\'PlayerP1\'); '
        'P1:hidden(0) end" />').root

    env, _warn = _run(root, [
        {1: 2.0, 2: 'Hide'}, {1: 6.0, 2: 'Show'}])
    timeline = modfile._base_field_hidden_timeline(env)
    assert timeline is not None
    assert timeline.sample(4.0)[0] == 1.0    # hidden between the two
    assert timeline.sample(8.0)[0] == 0.0    # shown after
    assert timeline.sample(0.0)[0] == 0.0    # rest: shown


def test_base_field_hidden_timeline_none_when_never_hidden():
    root = parse_actor_xml('<Quad InitCommand="x,0" />').root
    env, _warn = _run(root, [])
    assert modfile._base_field_hidden_timeline(env) is None


def test_aft_bg_visible_timeline_inverts_bg_quad_hidden():
    """A fullscreen bg quad toggled by ShowAFTBG/HideAFT compiles into a
    bg-visible timeline (bg in the AFT capture while the quad is shown)."""
    bg_stem = 'bg'
    root = parse_actor_xml(
        '<Quad File="bg.png" InitCommand="hidden,1" '
        'ShowAFTBGMessageCommand="hidden,0" '
        'HideAFTMessageCommand="hidden,1" />').root

    env, _warn = _run(root, [
        {1: 3.0, 2: 'ShowAFTBG'}, {1: 9.0, 2: 'HideAFT'}])
    actor_keyframes = env.actor_keyframes()
    timeline = modfile._aft_bg_visible_timeline(root, bg_stem, actor_keyframes)
    assert timeline is not None
    assert timeline.sample(1.0)[0] == 0.0    # rest hidden -> no bg
    assert timeline.sample(5.0)[0] == 1.0    # shown -> bg in capture
    assert timeline.sample(11.0)[0] == 0.0   # hidden again -> no bg


def test_aft_bg_visible_none_without_bg_quad():
    """A ShowAFTBG quad that does NOT draw the background image (a black
    Quad = ShowAFT-without-bg) contributes no bg-visible signal."""
    root = parse_actor_xml(
        '<Quad Type="Quad" InitCommand="hidden,1" '
        'ShowAFTMessageCommand="hidden,0" />').root
    env, _warn = _run(root, [{1: 3.0, 2: 'ShowAFT'}])
    assert modfile._aft_bg_visible_timeline(
        root, 'bg', env.actor_keyframes()) is None


def _run(root, mod_actions):
    """Load `root` under a stub env, install a mod_actions table of the
    given rows, and replay it (firing the broadcasts)."""
    env = StubEnvironment(start_beat=0.0, to_seconds=_seconds_identity)
    warnings = env.load_actors(root)
    lua_rows = ', '.join(
        "{%s, '%s'}" % (row[1], row[2]) for row in mod_actions)
    env.run('mod_actions = {%s}' % lua_rows, name='mod_actions')
    env.replay_mod_actions()
    return env, warnings
