from __future__ import annotations

from PySide6.QtWidgets import QFileDialog

from analysis.gui.widgets import HtmlTab
from analysis.viz.plots import generate_html_report
from analysis.gui.library.entry_actions.base import EntryActionBase


class HtmlReportAction(EntryActionBase):
    def run(self, entry: dict) -> None:
        if not entry:
            return

        out = self.save_path(f'report_{entry["game"]}.html')
        if not out:
            return

        title = self.title_song(entry, fallback='report')

        def job(progress):
            progress('parsing replay…')
            replay = self.tab._replay_cache.get(entry)

            progress('generating HTML report…')
            generate_html_report(replay, score_meta=entry, output_path=out)

            return out, title

        self.tab.jobs.run_dialog_job(
            title='HTML report',
            label=f'Generating report for {title}…',
            label_prefix=title,
            job=job,
            error_title='Failed to generate HTML report',
            on_done=self.finish,
        )

    def finish(self, payload) -> None:
        out, title = payload
        self.tab._add_tab(HtmlTab(out), f'📄 {title}')

    def save_path(self, default_name: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            'Save HTML',
            default_name,
            'HTML (*.html)',
        )
        return path or None