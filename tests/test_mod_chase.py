"""Chase compilation in render/mods/channels.py: the linear-segment form
of the engine's per-frame fapproach (PlayerOptions::Approach).

The regression these pin: a channel that re-targets faster than its
`*speed` chase can complete (beat/flip/mini toggling every ~0.1s in real
charts) must stay time-ordered. The old compiler emitted each ramp to
its full target at t + gap/speed with no clamp, so successive ramps
overran and made the breakpoint `times` non-monotonic - which broke the
bisect in `_sample` and produced runaway values (a `beat` percent
ramping to -1.34 where the engine holds near -0.1)."""
import numpy as np

from analysis.player.render.mods.channels import ModChannels, ModEvent


def _fapproach_reference(events, ts, speed_lookup=None):
    """The engine chase: current value moves toward the most-recent
    event's target by dt*speed each frame (fapproach, RageUtil.cpp:51 /
    PlayerOptions::Approach:40). Sampled finely to stand in for the
    per-frame engine loop."""
    ordered = sorted(events, key=lambda e: e.beat)
    dt = 1.0 / 1000.0
    cur = 0.0
    grid = np.arange(0.0, ts[-1] + dt, dt)
    vals = {}
    for t in grid:
        target, speed = 0.0, 1.0
        for ev in ordered:
            if ev.beat <= t:
                target, speed = ev.value, ev.speed
            else:
                break
        step = dt * (speed if speed > 0 else 1e9)
        if cur < target:
            cur = min(cur + step, target)
        elif cur > target:
            cur = max(cur - step, target)
        vals[round(t, 6)] = cur
    return [vals[min(vals, key=lambda k: abs(k - t))] for t in ts]


def _times(channels, mod, player=0):
    return channels._channels[(mod, player)].times


def test_fast_retarget_keeps_times_monotonic():
    """A channel toggled faster than its chase completes stays
    time-ordered (the bisect precondition `_sample` relies on)."""
    events = []
    for i in range(8):
        events.append(ModEvent(i * 0.5, 1.5, 1.0, 'beat'))
        events.append(ModEvent(i * 0.5 + 0.1, 0.0, 1.0, 'beat'))
    channels = ModChannels.compile(events)
    ts = _times(channels, 'beat')
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def test_interrupted_chase_matches_fapproach():
    """The compiled value tracks the per-frame engine chase through
    rapid re-targets - no runaway from overrunning ramps."""
    events = []
    for i in range(6):
        events.append(ModEvent(i * 0.5, 1.5, 1.0, 'beat'))
        events.append(ModEvent(i * 0.5 + 0.25, 0.0, 1.0, 'beat'))
    channels = ModChannels.compile(events)
    ts = np.arange(0.0, 3.0, 0.05)
    ours = [channels.value('beat', float(t)) for t in ts]
    ref = _fapproach_reference(events, ts)
    assert np.allclose(ours, ref, atol=2e-3)


def test_completed_chase_holds_target_until_retarget():
    """A chase that arrives before the next event holds its target across
    the gap (not a spurious ramp past it)."""
    channels = ModChannels.compile([
        ModEvent(0.0, 1.0, 2.0, 'drunk'),   # arrives at t=0.5
        ModEvent(2.0, 0.0, 2.0, 'drunk'),   # re-targets much later
    ])
    assert channels.value('drunk', 0.5) == 1.0
    assert channels.value('drunk', 1.0) == 1.0   # held, not overrun
    assert channels.value('drunk', 1.9) == 1.0


def test_slow_chase_reaches_partial_value_at_retarget():
    """When a re-target interrupts a chase mid-flight, the value carries
    forward from where it reached - the engine's stateful chase."""
    # speed 1: from 0 toward 1 over 1s, but re-targeted at t=0.4 (40%).
    channels = ModChannels.compile([
        ModEvent(0.0, 1.0, 1.0, 'mini'),
        ModEvent(0.4, 0.0, 1.0, 'mini'),
    ])
    assert channels.value('mini', 0.4) == np.float64(0.4)
    # then chases back down from 0.4 toward 0, arriving at t=0.8.
    assert channels.value('mini', 0.8) == 0.0
    assert abs(channels.value('mini', 0.6) - 0.2) < 1e-9


def test_snap_still_instant_between_chases():
    """A speed<=0 snap remains a vertical step even amid chases."""
    channels = ModChannels.compile([
        ModEvent(0.0, 0.5, 1.0, 'flip'),
        ModEvent(0.2, 1.0, -1.0, 'flip'),   # snap
    ])
    assert channels.value('flip', 0.199) < 0.5
    assert channels.value('flip', 0.2) == 1.0
