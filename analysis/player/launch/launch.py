from __future__ import annotations

import sys


def launch_from_replay(
    replay,
    game='etterna',
    od=8,
    bpms=None,
    sm_offset=0,
    audio_path=None,
    sv_sections=None,
    scroll_ms=400.0,
    scroll_mode=None,
    cmod_bpm=600.0,
    xmod_value=1.0,
    osu_speed=20,
):
    from PySide6.QtWidgets import QApplication
    from analysis.gui.player_tab import PlayerTab

    app = QApplication.instance() or QApplication(sys.argv[:1])
    tab = PlayerTab(
        replay,
        game=game,
        od=od,
        bpms=bpms,
        sm_offset=sm_offset,
        audio_path=audio_path,
        scroll_ms=scroll_ms,
        scroll_mode=scroll_mode,
        cmod_bpm=cmod_bpm,
        xmod_value=xmod_value,
        osu_speed=osu_speed,
    )
    tab.resize(1200, 900)
    tab.setWindowTitle('Replay Player')
    tab.show()
    return app.exec()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print("usage: player.py <replay> [--osu chart.osu] [--audio file] [--od N]")
        return 1

    path = argv[0]
    od = float(argv[argv.index('--od') + 1]) if '--od' in argv else 8

    from analysis.core.game import resolve_standalone_replay

    game, rep, bpms, sm_off, audio, _extra = resolve_standalone_replay(
        path,
        args=argv,
    )
    return launch_from_replay(
        rep,
        game=game,
        od=od,
        bpms=bpms,
        sm_offset=sm_off,
        audio_path=audio,
    )


if __name__ == '__main__':
    raise SystemExit(main())
