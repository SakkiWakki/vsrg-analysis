"""Fakes (SM character 'F', Etterna TapNoteType_Fake).

Render normally but are excluded from judgment ; you can't hit or miss
them. Etterna filters fakes out of the scored note list at chart-load
time. Our fingerprint deliberately excludes them
(analysis.games.etterna.sm_chart._chart_fingerprint) so replay matching
still works on charts with fake-rich intros."""
from __future__ import annotations

from analysis.player.notetypes import NT_FAKE  # noqa: F401
