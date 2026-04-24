from __future__ import annotations

from pathlib import Path

from analysis.core import game as game_mod
from analysis.core import gui_adapter as gui_mod


class EntryActionBase:
    def __init__(self, tab):
        self.tab = tab

    def title_song(self, entry: dict, *, fallback: str = 'entry') -> str:
        return (
            entry.get('song')
            or Path(entry.get('replay_path') or '').name
            or fallback
        )[:40]

    def maybe_backfill_entry(self, entry: dict, replay: dict) -> None:
        if replay.get('chart_path') and not entry.get('chart_path'):
            entry['chart_path'] = replay['chart_path']

        if gui_mod.get(entry['game']).enrich_entry(entry):
            self.persist_library()
            self.tab.refresh_tree()

    def persist_library(self) -> None:
        for adapter in game_mod.all_games().values():
            try:
                adapter.save_cached(self.tab.library)
            except Exception:
                pass