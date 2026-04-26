from __future__ import annotations

import os
import threading
import time

from analysis.core import game as game_mod
from analysis.core import manifest as manifest_mod
from analysis.gui.player_tab import PlayerTab
from analysis.gui.settings import get_settings, load_player_settings

from analysis.gui.library.entry_actions.base import EntryActionBase


_STALL = os.environ.get('VSRG_STALL_DEBUG', '0') not in ('', '0', 'false')


def _slog(msg: str) -> None:
    if not _STALL:
        return
    tname = threading.current_thread().name
    print(f'[stall                  {tname}] {msg}', flush=True)


class PlayReplayAction(EntryActionBase):
    def run(self, entry: dict, *, replay=None) -> None:
        _slog(f'PlayReplayAction.run: game={entry.get("game")} '
              f'song={(entry.get("song") or "")[:40]!r} '
              f'chart_path={"yes" if entry.get("chart_path") else "no"}')
        default_ms = self.default_scroll_ms()
        get_settings().setValue('library/default_scroll_ms', default_ms)

        scroll_mode = load_player_settings(entry['game'])['scroll_mode']
        title_song = self.title_song(entry)

        def job(progress):
            _slog('  job: parsing replay…')
            t0 = time.monotonic()
            progress('parsing replay…')
            loaded = replay if replay is not None else self.tab._replay_cache.get(entry)
            _slog(f'  job: replay parsed ({(time.monotonic()-t0)*1000:.1f}ms)')

            _slog('  job: resolving chart/audio…')
            t0 = time.monotonic()
            progress('resolving chart/audio…')
            bpms, sm_off, audio = manifest_mod.get(entry['game']).resolve_chart_context(
                loaded,
                entry=entry,
                progress=progress,
            )
            _slog(f'  job: chart/audio resolved ({(time.monotonic()-t0)*1000:.1f}ms) audio={bool(audio)}')
            return loaded, bpms, sm_off, audio

        self.tab.jobs.run_dialog_job(
            title='Replay',
            label=f'Loading {title_song}…',
            label_prefix=title_song,
            job=job,
            error_title='Failed to load replay',
            on_done=lambda payload: self.finish(
                entry,
                payload,
                default_ms=default_ms,
                scroll_mode=scroll_mode,
            ),
        )

    def finish(
        self,
        entry: dict,
        payload,
        *,
        default_ms: float,
        scroll_mode: str,
    ) -> None:
        _slog('PlayReplayAction.finish enter (main thread)')
        replay, bpms, sm_off, audio = payload
        t0 = time.monotonic()
        self.maybe_backfill_entry(entry, replay)
        _slog(f'  maybe_backfill_entry done ({(time.monotonic()-t0)*1000:.1f}ms)')

        rate = float(entry.get('rate') or 1.0)
        t0 = time.monotonic()
        extra = game_mod.get(entry['game']).player_tab_kwargs(
            replay,
            entry,
            (bpms, sm_off, audio),
        )
        _slog(f'  player_tab_kwargs done ({(time.monotonic()-t0)*1000:.1f}ms)')
        scroll_mode = extra.pop('scroll_mode', scroll_mode)

        _slog('  -> PlayerTab(...)')
        t0 = time.monotonic()
        tab = PlayerTab(
            replay,
            game=entry['game'],
            audio_path=audio,
            scroll_ms=default_ms,
            scroll_mode=scroll_mode,
            play_rate=rate,
            **extra,
        )
        _slog(f'  <- PlayerTab() returned ({(time.monotonic()-t0)*1000:.1f}ms)')

        title = self.title_song(entry, fallback='play')
        _slog('  -> _add_tab')
        t0 = time.monotonic()
        self.tab._add_tab(tab, f'▶ {title}')
        _slog(f'  <- _add_tab done ({(time.monotonic()-t0)*1000:.1f}ms)')
        _slog('PlayReplayAction.finish exit')

    def default_scroll_ms(self) -> float:
        try:
            return float(self.tab.default_scroll_edit.text().strip() or 400)
        except ValueError:
            return 400.0
