from __future__ import annotations

from pathlib import Path

from analysis.core import game as game_mod
from analysis.core import manifest as manifest_mod


class EntryActionBase:
    def __init__(self, tab):
        self.tab = tab

    def tab_key(self, kind: str, entry: dict, *extra):
        """Identity of the tab this action would open, or None (no
        dedup) when the entry lacks a stable identity. Actions check
        ``self.tab.focus_tab(key)`` before running their load job and
        pass the same key to ``_add_tab`` so re-invoking the action
        jumps to the open tab instead of building a duplicate."""
        replay_path = entry.get('replay_path')
        if not replay_path:
            return None
        return (kind, replay_path, *extra)

    def title_song(self, entry: dict, *, fallback: str = 'entry') -> str:
        return (
            entry.get('song')
            or Path(entry.get('replay_path') or '').name
            or fallback
        )[:40]

    def maybe_backfill_entry(self, entry: dict, replay: dict) -> None:
        if replay.get('chart_path') and not entry.get('chart_path'):
            entry['chart_path'] = replay['chart_path']

        if manifest_mod.get(entry['game']).enrich_entry(entry):
            self.persist_library()
            self.tab.refresh_tree()

    def persist_library(self) -> None:
        for adapter in game_mod.all_games().values():
            try:
                adapter.save_cached(self.tab.library)
            except Exception:
                pass