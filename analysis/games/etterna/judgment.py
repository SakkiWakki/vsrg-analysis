"""Etterna judge system. J1..J8 + J9/JUSTICE.

Tap windows (marv/perf/great/good) scale linearly from the J4 baseline.
Bad is also scaled for J1..J4 but clamps at 180 ms from J5 down (the
wiki table footnotes this, and Etterna's code freezes it there). Hold,
Roll, and Mine windows behave the same way ; they scale with judge 1..4
and freeze at the J4 value for harder judges.

Exposed through `EtternaAdapter.judgement_windows(replay, judge='J4')`.
The viz layer pulls `windows_for` directly when drawing window bands."""
from __future__ import annotations


# Tap windows (half-window, seconds) at J4. These scale linearly with
# ETT_JUDGE_SCALES for every judge.
_TAP_J4 = [
    ('marv', 0.0225),
    ('perf', 0.045),
    ('great', 0.090),
    ('good', 0.135),
]

# Bad / hold / roll / mine at J4. Scale with judges 1..4; freeze here
# at J5 and harder.
_BAD_J4 = 0.180
_HOLD_J4 = 0.250
_ROLL_J4 = 0.500
_MINE_J4 = 0.075

# Scale factor relative to J4
ETT_JUDGE_SCALES = {
    'J1': 1.50, 'J2': 1.33, 'J3': 1.16, 'J4': 1.00,
    'J5': 0.84, 'J6': 0.66, 'J7': 0.50, 'J8': 0.33,
    'J9': 0.20, 'JUSTICE': 0.20,
}


def windows_for(judge='J4'):
    """Return [(name, half_window_sec), …] for the given judge.
    Includes tap windows (marv..good) always, plus Bad which freezes at
    J4's 180 ms for harder judges. Hold/roll/mine are available via
    `extra_windows_for` since the player's tap-judge path doesn't use
    them yet."""
    scale = ETT_JUDGE_SCALES.get(str(judge).upper(), 1.0)
    windows = [(n, w * scale) for (n, w) in _TAP_J4]
    # Bad clamps ; scale down for easy judges, hold flat for hard ones.
    windows.append(('bad', _BAD_J4 * min(1.0, scale)))
    return windows


def extra_windows_for(judge='J4'):
    """Hold / roll / mine tolerance in seconds. Same clamp rule as bad ;
    scales only for J1..J4 where the table still moves, then freezes."""
    scale = ETT_JUDGE_SCALES.get(str(judge).upper(), 1.0)
    clamp = min(1.0, scale)
    return {
        'hold': _HOLD_J4 * clamp,
        'roll': _ROLL_J4 * clamp,
        'mine': _MINE_J4 * clamp,
    }


# Back-compat export used by the viz layer for the window-band plot.
# Matches the `windows_for` default-judge output in shape so existing
# code can keep iterating `(name, sec)` pairs.
JUDGE_WINDOWS_ETT_J4 = _TAP_J4 + [('bad', _BAD_J4)]
