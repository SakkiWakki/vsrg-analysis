"""Etterna game adapter + scroll modes (CMOD + XMOD)."""
from __future__ import annotations

from pathlib import Path

from analysis.core.game import GameAdapter
from analysis.player import scroll


class EtternaAdapter(GameAdapter):
    name = 'etterna'

    def parse_replay(self, path, chart_path=None):
        from analysis.games.etterna.replay import parse_replay
        return parse_replay(path)

    def _find_chart(self, replay, entry=None, progress=None):
        from analysis.games.etterna.replay import find_etterna_dirs
        from analysis.games.etterna.sm_chart import (find_chart_by_key,
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
                hit = find_chart_by_key(chartkey, songs, progress=progress)
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
        from analysis.games.etterna.judgment import windows_for
        return windows_for(judge or 'J4')

    def nudge_judge(self, current, delta):
        """Step through J1..J9 by one per click. Etterna's judge is
        discrete — integers only — so we take the sign of `delta` and
        clamp to [1, 9]."""
        cur = str(current or 'J4').upper()
        if cur == 'JUSTICE':
            cur = 'J9'
        try:
            n = int(cur.lstrip('J'))
        except ValueError:
            n = 4
        step = 1 if delta > 0 else (-1 if delta < 0 else 0)
        n = max(1, min(9, n + step))
        return f'J{n}'

    def prepare_replay_times(self, replay, bpms=None, sm_offset=0.0, **_):
        import numpy as np
        from analysis.games.etterna.sm_chart import row_to_time
        from analysis.player.timing import infer_keycount
        if bpms is not None:
            times = np.array([row_to_time(int(r), bpms, sm_offset)
                              for r in replay['noterows']])
        else:
            # 120bpm, 48 rows/beat => 96 rows/sec
            times = replay['noterows'].astype(np.float64) / 96.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                if bpms is not None:
                    hold_tails[(h[0], h[1])] = row_to_time(
                        int(h[2]), bpms, sm_offset)
                else:
                    hold_tails[(h[0], h[1])] = h[2] / 96.0
        return times, hold_tails, infer_keycount(replay)

    def judge_label(self, replay, judge=None, **_):
        return str(judge or 'J4')

    def default_scroll_mode(self):
        return 'cmod'

    def player_kwargs(self, replay, judge=None, **_):
        return {'ett_judge': judge or 'J4'}


# --- Etterna scroll modes ----------------------------------------------------
# Values expressed in the 480-tall Til Death / fallback theme space; scaled to
# window H in the Player. ArrowSpacing=64 comes from Themes/_fallback/metrics.
_ARROW_SPACING = 64.0


def _cmod_to_pxps(value, opts, p):
    # Etterna CMOD: Y = secondsUntilNote * (bpm/60) * ArrowSpacing
    # (ArrowEffects.cpp:358-372). Mini mod scales the NoteField
    # (Player.cpp:810: 1 - mini * 0.5).
    field_scale = p.H / p.REFERENCE_FIELD_H
    mini = float(opts.get('mini', 0.0))
    zoom = max(0.05, 1.0 - mini * 0.5)
    return ((float(value) / 60.0) * _ARROW_SPACING * field_scale * zoom
            * float(opts.get('receptor_size', 1.0)))


def _cmod_from_pxps(pxps, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    mini = float(opts.get('mini', 0.0))
    zoom = max(0.05, 1.0 - mini * 0.5)
    denom = (_ARROW_SPACING * field_scale * zoom
             * float(opts.get('receptor_size', 1.0)))
    return pxps * 60.0 / max(1e-9, denom)


def _cmod_on_enter(player, state):
    """CMOD ignores SV (ArrowEffects.cpp only applies SV in the XMOD branch).
    Remember prior SV-enabled state in this mode's state dict so that
    switching away restores exactly what the user had before."""
    state['sv_enabled_saved'] = player.sv_enabled
    player.sv_enabled = False


def _cmod_on_exit(player, state):
    if 'sv_enabled_saved' in state:
        player.sv_enabled = state.pop('sv_enabled_saved')


def _cmod_fmt(v):
    iv = int(round(v))
    return f'C{iv}' if abs(v - iv) < 1e-4 else f'C{v:.2f}'


scroll.register(scroll.ScrollMode(
    key='cmod',
    label='CMOD',
    game='etterna',
    to_pxps=_cmod_to_pxps,
    from_pxps=_cmod_from_pxps,
    default_value=600.0,
    value_bounds=(60.0, 5000.0),
    nudge=scroll.multiplicative_nudge,
    format_value=_cmod_fmt,
    options={'mini': 0.0, 'receptor_size': 1.0},
    on_enter=_cmod_on_enter,
    on_exit=_cmod_on_exit,
))


def _xmod_to_pxps(value, opts, p):
    """XMOD: beat-spacing. Etterna's XMOD branch (ArrowEffects.cpp) renders
    each beat as ArrowSpacing * xmod_value pixels, independent of BPM. We
    express the scalar as the multiplier so xmod=1.0 → 64 px/beat at the
    reference field height. Respects mini via NoteField zoom; no receptor
    scaling here — the user said 'if something exists for CMOD but not for
    XMOD, don't implement it for XMOD as well'. SV layers on top (XMOD is
    the branch where GetDisplayedSpeedPercent actually applies SV).

    To convert beat-rate into time-rate for the Player's px/sec contract,
    we need a representative BPM. We use the replay's average BPM if known
    (precomputed on the player), else 120."""
    field_scale = p.H / p.REFERENCE_FIELD_H
    mini = float(opts.get('mini', 0.0))
    zoom = max(0.05, 1.0 - mini * 0.5)
    bpm = float(getattr(p, '_xmod_reference_bpm', 120.0))
    return (float(value) * (bpm / 60.0) * _ARROW_SPACING * field_scale * zoom)


def _xmod_from_pxps(pxps, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    mini = float(opts.get('mini', 0.0))
    zoom = max(0.05, 1.0 - mini * 0.5)
    bpm = float(getattr(p, '_xmod_reference_bpm', 120.0))
    denom = (bpm / 60.0) * _ARROW_SPACING * field_scale * zoom
    return pxps / max(1e-9, denom)


def _xmod_fmt(v):
    return f'X{v:.2f}'


scroll.register(scroll.ScrollMode(
    key='xmod',
    label='XMOD',
    game='etterna',
    to_pxps=_xmod_to_pxps,
    from_pxps=_xmod_from_pxps,
    default_value=1.0,
    value_bounds=(0.1, 20.0),
    nudge=scroll.multiplicative_nudge,
    format_value=_xmod_fmt,
    options={'mini': 0.0},
))


ADAPTER = EtternaAdapter()
