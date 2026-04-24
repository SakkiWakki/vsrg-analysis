"""Compatibility shim for the old analysis.player.player import path."""

from analysis.player.player_api import (
    Player,
    etterna_windows_for,
    osu_mania_windows,
    prepare_replay_times,
    launch_from_replay,
)

__all__ = [
    'Player',
    'etterna_windows_for',
    'osu_mania_windows',
    'prepare_replay_times',
    'launch_from_replay',
]


if __name__ == '__main__':
    from analysis.player.launch import main
    raise SystemExit(main())
