"""Specs for the render-thread sweep nudge's frame budget.

The nudge borrows render-thread time to advance the lazy sweep. Its budget is
a share of what is LEFT of the frame, not a constant, because a constant
spends the same milliseconds on a machine with 11ms of headroom and one with
none - and the machine with none is the one whose frames the sweep would
break. These specs pin the shape across hardware we cannot run here: full
budget where there is room, back off where there is not, never zero - and
past the point where the frame is lost outright, escalate: those frames are
slow BECAUSE the sweep is unfinished (the legacy path draws until the doc
adopts, and the worker parks while frames arrive), so a floor there
preserves the slow regime forever instead of protecting anything.
"""
from analysis.games.notitg.sim.producers import (
    _NUDGE_CAP_S, _NUDGE_FLOOR_S, _NUDGE_FRAME_S, _NUDGE_LOST_CAP_S,
    _NUDGE_LOST_X, _LiveFieldInstances)

_budget = _LiveFieldInstances._nudge_budget


class _Nudger:
    """The two fields `_nudge_budget` reads, without building a whole sim."""

    def __init__(self, spent: float = 0.0):
        self._nudge_spent = spent


def _settled(render_s: float) -> float:
    """The budget the controller converges on when everything except the
    nudge costs `render_s` per frame. The controller sees the WHOLE interval,
    so its own slice feeds back into the next frame's measurement."""
    nudger = _Nudger()
    budget = 0.0
    for _ in range(32):
        budget = _budget(nudger, render_s + nudger._nudge_spent)
        nudger._nudge_spent = budget
    return budget


def test_no_measurable_frame_yet_gets_the_full_cap():
    """First call, and anything after a pause/menu, has no usable interval:
    fall back to the cap rather than infer a budget from a nonsense reading."""
    assert _budget(_Nudger(), 0.0) == _NUDGE_CAP_S
    assert _budget(_Nudger(), -1.0) == _NUDGE_CAP_S
    assert _budget(_Nudger(), 5.0) == _NUDGE_CAP_S


def test_machine_with_headroom_keeps_the_full_slice():
    """A fast machine must not be penalised by the controller: with most of
    the frame spare, the budget saturates at the cap exactly as the old fixed
    budget did."""
    assert _settled(0.004) == _NUDGE_CAP_S
    assert _settled(0.009) == _NUDGE_CAP_S


def test_budget_shrinks_as_the_frame_fills():
    """Between "room to spare" and "no room", the budget falls monotonically
    with the cost of everything else."""
    budgets = [_settled(ms / 1000.0) for ms in (4, 9, 12, 14, 16)]
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[0] == _NUDGE_CAP_S
    assert budgets[-1] < budgets[0]


def test_frame_stays_within_target_while_any_headroom_remains():
    """The point of the controller: while the frame is not already lost, the
    nudge must not be what loses it."""
    for ms in (4, 9, 12, 14, 15):
        render_s = ms / 1000.0
        assert render_s + _settled(render_s) <= _NUDGE_FRAME_S


def test_marginally_missing_frames_fall_to_the_floor():
    """A machine near the target gets the smallest slice we still make
    progress on - a marginal miss must not be deepened by the sweep."""
    for ms in (17, 25, 30):
        assert _settled(ms / 1000.0) == _NUDGE_FLOOR_S


def test_lost_frames_escalate_to_end_the_regime():
    """Past `_NUDGE_LOST_X` times the target the frame is not salvageable,
    and it is slow BECAUSE the sweep is unfinished - the budget becomes a
    share of the whole interval so the regime ends in about a minute
    instead of persisting for hours at the floor (gat 2's 'background
    compile' never finishing while the chart played)."""
    lost = _settled(0.150)
    assert lost > 10 * _NUDGE_FLOOR_S
    assert lost <= _NUDGE_LOST_CAP_S
    assert _settled(0.300) == _NUDGE_LOST_CAP_S
    # The regimes meet monotonically: a frame just past the lost threshold
    # never gets LESS than the floor band it left.
    just_lost = _settled(_NUDGE_LOST_X * _NUDGE_FRAME_S + 0.002)
    assert just_lost >= _NUDGE_FLOOR_S


def test_budget_is_never_zero():
    """A zero budget would stall the frontier for as long as frames keep
    arriving, since the background worker parks while rendering is live."""
    for ms in range(0, 120, 3):
        assert _settled(ms / 1000.0) >= _NUDGE_FLOOR_S > 0.0


def test_budget_never_exceeds_the_cap():
    """Even an implausibly cheap frame cannot hand the sweep an unbounded
    slice - the caps bound lock hold time (the healthy cap on healthy
    frames, the escalation cap on lost ones)."""
    for ms in range(0, 20):
        assert _settled(ms / 1000.0) <= _NUDGE_CAP_S
    for ms in range(40, 400, 20):
        assert _settled(ms / 1000.0) <= _NUDGE_LOST_CAP_S
