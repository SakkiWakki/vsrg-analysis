"""Per-game adapter: audio resolution, chart timing, judgment windows.

Centralizes the `game == 'osu' / 'etterna'` branches scattered across the
loaders, player, and viz code. Callers pull an adapter via `get(name)` and
use its methods uniformly.
"""
from __future__ import annotations
from pathlib import Path


class GameAdapter:
    name: str = ''

    def parse_replay(self, path, chart_path=None):
        raise NotImplementedError

    def resolve_audio(self, replay, entry=None, progress=None) -> str | None:
        return None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        """Return (bpms, sm_offset). Both may be None/0.0 if not applicable."""
        return None, 0.0

    def judgement_windows(self, replay, **overrides):
        raise NotImplementedError

    def judge_label(self, replay, **overrides) -> str:
        raise NotImplementedError

    def default_scroll_mode(self) -> str:
        return 'linear'

    def player_kwargs(self, replay, **overrides) -> dict:
        """Extra kwargs the Player __init__ needs for this game
        (OD for osu, judge for etterna)."""
        return {}


class OsuAdapter(GameAdapter):
    name = 'osu'

    def parse_replay(self, path, chart_path=None):
        from analysis.osu.replay import parse_replay, find_osu_dirs
        songs = find_osu_dirs().get('songs_dir')
        return parse_replay(path, osu_path=chart_path, songs_dir=songs)

    def resolve_audio(self, replay, entry=None, progress=None):
        from analysis.osu.replay import parse_osu_file
        if not replay.get('chart_path'):
            return None
        chart = parse_osu_file(replay['chart_path'])
        audio = chart.get('audio')
        if not audio:
            return None
        cand = Path(replay['chart_path']).parent / audio
        return str(cand) if cand.exists() else None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        # osu! replays carry absolute ms timings; no sm-style offset needed.
        return None, 0.0

    def effective_od(self, replay, od=None):
        from analysis.viz.note_visualizer import effective_osu_od
        base = od if od is not None else float(replay.get('od', 8.0))
        mods = int(replay.get('mods', 0))
        return effective_osu_od(base, mods)

    def judgement_windows(self, replay, od=None, **_):
        from analysis.player.player import osu_mania_windows
        return osu_mania_windows(self.effective_od(replay, od))

    def judge_label(self, replay, od=None, **_):
        return f'OD {self.effective_od(replay, od):g}'

    def default_scroll_mode(self):
        return 'linear'

    def player_kwargs(self, replay, od=None, **_):
        return {'od': self.effective_od(replay, od)}


class EtternaAdapter(GameAdapter):
    name = 'etterna'

    def parse_replay(self, path, chart_path=None):
        from analysis.etterna.replay import parse_replay
        return parse_replay(path)

    def _find_chart(self, replay, entry=None, progress=None):
        from analysis.etterna.replay import find_etterna_dirs
        from analysis.etterna.sm_chart import (find_chart_by_key,
                                            find_chart_for_replay)
        save = find_etterna_dirs().get('save_dir')
        if not save:
            return None
        songs = Path(save).parent / 'Songs'
        if not songs.exists():
            return None
        chartkey = (entry or {}).get('chart_key')
        if chartkey:
            try:
                if progress:
                    progress('chartkey lookup…')
                hit = find_chart_by_key(chartkey, songs)
                if hit:
                    return hit
            except Exception:
                pass
        try:
            if progress:
                progress('fingerprint chart match…')
            return find_chart_for_replay(replay['noterows'],
                                         replay['columns'], songs)
        except Exception:
            return None

    def resolve_audio(self, replay, entry=None, progress=None):
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None
        music = found['data'].get('music', '')
        if not music:
            return None
        cand = Path(found['file']).parent / music
        return str(cand) if cand.exists() else None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None, 0.0
        return found['data']['bpms'], found['data']['offset']

    def resolve_all(self, replay, entry=None, progress=None):
        """Single-pass combined resolver — avoids parsing the .sm/.ssc twice.
        Returns (bpms, offset, audio_path)."""
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None, 0.0, None
        bpms = found['data']['bpms']
        offset = found['data']['offset']
        audio = None
        music = found['data'].get('music', '')
        if music:
            cand = Path(found['file']).parent / music
            if cand.exists():
                audio = str(cand)
        return bpms, offset, audio

    def judgement_windows(self, replay, judge=None, **_):
        from analysis.player.player import etterna_windows_for
        return etterna_windows_for(judge or 'J4')

    def judge_label(self, replay, judge=None, **_):
        return str(judge or 'J4')

    def default_scroll_mode(self):
        return 'cmod'

    def player_kwargs(self, replay, judge=None, **_):
        return {'ett_judge': judge or 'J4'}


_REGISTRY: dict[str, GameAdapter] = {
    'osu': OsuAdapter(),
    'etterna': EtternaAdapter(),
}


def get(name: str) -> GameAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown game: {name!r}')
