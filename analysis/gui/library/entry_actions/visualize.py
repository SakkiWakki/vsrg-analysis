from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox

from analysis.gui.widgets import MplTab, _viz_toolbar
from analysis.gui.library.entry_actions.base import EntryActionBase


class OpenVisualizationAction(EntryActionBase):
    def __init__(self, tab, play_action):
        super().__init__(tab)
        self.play_action = play_action

    def run(self, entry: dict, *, viz_name: str | None = None) -> None:
        if not entry:
            return

        match = self.find_visualization(viz_name)
        if match is None:
            QMessageBox.warning(self.tab, 'unknown visualization', viz_name or '')
            return

        name, builder, category = match
        title_song = self.title_song(entry)
        tab_title = f'📊 {name} ; {title_song}'

        def job(progress):
            progress('parsing replay…')
            replay = self.tab._replay_cache.get(entry)

            if category == 'chart':
                progress(f'rendering {name}…')
                prebuilt = builder(
                    replay,
                    game=entry['game'],
                    entry=entry,
                )
                return 'chart', replay, prebuilt

            return 'widget', replay, None

        self.tab.jobs.run_dialog_job(
            title=name,
            label=f'Processing {title_song}…',
            label_prefix=title_song,
            job=job,
            error_title=f'Failed to load {name}',
            on_done=lambda payload: self.finish(
                entry,
                payload,
                builder=builder,
                tab_title=tab_title,
            ),
        )

    def find_visualization(self, viz_name: str | None):
        import analysis.viz.plugins as viz_pkg

        name = viz_name or self.tab.viz_cb.currentText()
        return next(
            (
                (vn, builder, category)
                for vn, builder, category in viz_pkg.all_visualizations()
                if vn == name
            ),
            None,
        )

    def finish(self, entry: dict, payload, *, builder, tab_title: str) -> None:
        kind, replay, prebuilt = payload
        self.maybe_backfill_entry(entry, replay)

        on_play = lambda e=entry, r=replay: self.play_action.run(e, replay=r)

        if kind == 'chart':
            widget = MplTab(prebuilt, on_play=on_play)
        else:
            widget = self.build_widget_viz(builder, replay, entry, on_play)

        self.tab._add_tab(widget, tab_title)

    def build_widget_viz(self, builder, replay, entry, on_play):
        try:
            result = builder(
                replay,
                game=entry['game'],
                on_play=on_play,
                entry=entry,
            )
        except TypeError:
            result = builder(replay, game=entry['game'], entry=entry)

        if not isinstance(result, QWidget):
            return result

        if getattr(result, '_has_play_btn', False):
            return result

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(result, 1)
        layout.addLayout(_viz_toolbar(on_play))
        return wrapper