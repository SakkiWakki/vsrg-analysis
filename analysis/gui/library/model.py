from __future__ import annotations

from dataclasses import dataclass

from analysis.core.search import search


@dataclass(frozen=True)
class EntryRow:
    entry: dict
    values: list[str]


@dataclass(frozen=True)
class GroupRow:
    parent: EntryRow
    children: list[EntryRow]


def entry_values(e: dict) -> list[str]:
    return [
        e['game'],
        str(e.get('keycount') or '?'),
        (e.get('song') or '')[:200],
        (e.get('pack') or '')[:80],
        e.get('steps', '') or '',
        f"{e.get('rate', 1):.2f}",
        f"{e.get('wife', 0) * 100:.2f}",
        e.get('grade', '') or '',
        (e.get('datetime') or '')[:19],
    ]


class LibraryQuery:
    def __init__(self, tab):
        self.tab = tab
        self.library: list[dict] = []

    def set_library(self, library: list[dict]) -> None:
        self.library = library

    def rows(self) -> list[EntryRow | GroupRow]:
        if not self.library:
            return []

        min_wife = self._min_wife()
        game = self.tab.game_cb.currentText()
        key_filter = self._key_filter()

        results = search(
            self.library,
            query=self.tab.filter_edit.text() or None,
            game=None if game == 'all' else game,
            min_wife=min_wife,
            sort=self.tab.sort_cb.currentText(),
            descending=self.tab.desc_cbx.isChecked(),
            limit=None,
        )

        if key_filter is not None:
            results = [e for e in results if e.get('keycount') == key_filter]

        if not self.tab.group_cbx.isChecked():
            return [EntryRow(e, entry_values(e)) for e in results[:5000]]

        return self._grouped_rows(results, max_rows=5000)

    def _min_wife(self) -> float:
        try:
            return float(self.tab.min_wife_edit.text()) / 100.0
        except ValueError:
            return 0.0

    def _key_filter(self) -> int | None:
        text = self.tab.keys_cb.currentText().strip()
        try:
            value = int(text)
            return value if value > 0 else None
        except ValueError:
            return None

    def _grouped_rows(self, entries: list[dict], *, max_rows: int):
        groups: dict[tuple[str, str], list[dict]] = {}
        order: list[tuple[str, str]] = []

        for e in entries:
            key = (e['game'], (e.get('song') or '').strip())
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(e)

        rows: list[EntryRow | GroupRow] = []
        shown = 0

        for key in order:
            if shown >= max_rows:
                break

            children = groups[key]
            if len(children) == 1:
                e = children[0]
                rows.append(EntryRow(e, entry_values(e)))
                shown += 1
                continue

            best = max(children, key=lambda x: x.get('wife', 0))
            parent_values = entry_values(best)
            parent_values[2] = f'[{len(children)}]  {parent_values[2]}'

            rows.append(
                GroupRow(
                    parent=EntryRow(best, parent_values),
                    children=[EntryRow(e, entry_values(e)) for e in children],
                )
            )
            shown += 1 + len(children)

        return rows