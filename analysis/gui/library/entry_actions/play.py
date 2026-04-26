from __future__ import annotations

from analysis.core import game as game_mod
from analysis.core import manifest as manifest_mod
from analysis.gui.player_tab import PlayerTab
from analysis.gui.settings import get_settings, load_player_settings

from analysis.gui.library.entry_actions.base import EntryActionBase


class PlayReplayAction(EntryActionBase):
    def run(self, entry: dict, *, replay=None) -> None:
        default_ms = self.default_scroll_ms()
        get_settings().setValue('library/default_scroll_ms', default_ms)

        scroll_mode = load_player_settings(entry['game'])['scroll_mode']
        title_song = self.title_song(entry)

        def job(progress):
            progress('parsing replay…')
            loaded = replay if replay is not None else self.tab._replay_cache.get(entry)

            progress('resolving chart/audio…')
            bpms, sm_off, audio = manifest_mod.get(entry['game']).resolve_chart_context(
                loaded,
                entry=entry,
                progress=progress,
            )
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
        replay, bpms, sm_off, audio = payload
        self.maybe_backfill_entry(entry, replay)

        rate = float(entry.get('rate') or 1.0)
        extra = game_mod.get(entry['game']).player_tab_kwargs(
            replay,
            entry,
            (bpms, sm_off, audio),
        )
        scroll_mode = extra.pop('scroll_mode', scroll_mode)

        tab = PlayerTab(
            replay,
            game=entry['game'],
            audio_path=audio,
            scroll_ms=default_ms,
            scroll_mode=scroll_mode,
            play_rate=rate,
            **extra,
        )

        title = self.title_song(entry, fallback='play')
        self.tab._add_tab(tab, f'▶ {title}')

    def default_scroll_ms(self) -> float:
        try:
            return float(self.tab.default_scroll_edit.text().strip() or 400)
        except ValueError:
            return 400.0
