from __future__ import annotations

from analysis.gui.library.entry_actions import (
    PlayReplayAction,
    OpenVisualizationAction,
    HtmlReportAction,
)

class LibraryOpeners:
    def __init__(self, tab):
        self.tab = tab

        play = PlayReplayAction(tab)
        self.actions = {
            'play': play,
            'visualize': OpenVisualizationAction(tab, play),
            'html_report': HtmlReportAction(tab),
        }

    def play_selected(self) -> None:
        if entry := self.tab.selected_entry():
            self.run('play', entry)

    def run(self, name: str, entry: dict, **kwargs) -> None:
        self.actions[name].run(entry, **kwargs)