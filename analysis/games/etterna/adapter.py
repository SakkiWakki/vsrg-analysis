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
        return self._resolve_music_asset(found['file'], music)

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None, 0.0
        return found['data']['bpms'], found['data']['offset']

    def resolve_all(self, replay, entry=None, progress=None):
        """Single-pass combined resolver — avoids parsing the .sm/.ssc twice.
        Returns (bpms, offset, audio_path).

        Side effect: enriches `replay['holds']` from 2-tuples `(head_row,
        col)` (all the .bin replay file records) into 3-tuples `(head_row,
        col, end_row)` by joining with the matched chart's hold spans. The
        LN renderer relies on the 3-tuple shape to draw tails."""
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None, 0.0, None
        bpms = found['data']['bpms']
        offset = found['data']['offset']
        audio = None
        music = found['data'].get('music', '')
        if music:
            audio = self._resolve_music_asset(found['file'], music)
        self._attach_chart_hold_ends(replay, found)
        self._attach_chart_extras(replay, found)
        return bpms, offset, audio

    @staticmethod
    def _resolve_music_asset(chart_file, music):
        """Resolve #MUSIC the way Etterna does, returning an OS path.

        Etterna first applies Song::GetSongAssetPath. If that does not point
        at a file, Song::TidyUpData scans the song folder and uses the first
        sound file it finds.
        """
        resolved = EtternaAdapter._resolve_song_asset(chart_file, music)
        if resolved:
            return resolved
        song_dir = Path(chart_file).parent
        try:
            files = sorted(
                (p for p in song_dir.iterdir() if p.is_file()),
                key=lambda p: p.name.casefold(),
            )
        except OSError:
            return None
        for p in files:
            if p.suffix[1:].casefold() in {'mp3', 'oga', 'ogg', 'wav'}:
                return str(p)
        return None

    @staticmethod
    def _resolve_song_asset(chart_file, asset):
        """Resolve a simfile asset path using Etterna's asset path rules.

        Etterna's FilenameDB resolves paths case-insensitively and replaces
        them with the real on-disk case before opening. We mirror that final
        canonicalization so Python decoders receive a path that exists.
        """
        candidate = EtternaAdapter._song_asset_path(chart_file, asset)
        if candidate is None:
            return None
        ci = EtternaAdapter._case_insensitive_existing_path(candidate)
        return str(ci) if ci is not None else None

    @staticmethod
    def _song_asset_path(chart_file, asset):
        if not asset:
            return None
        asset = str(asset).strip().replace('\\', '/')
        if not asset:
            return None
        base = Path(chart_file).parent
        rel_candidate = base / asset
        if EtternaAdapter._case_insensitive_existing_path(rel_candidate) is not None:
            return rel_candidate
        if '/' not in asset:
            return rel_candidate

        root = EtternaAdapter._simfile_root_for_chart(chart_file)
        if asset.startswith('../'):
            candidate = rel_candidate
        else:
            # Etterna treats paths with slashes as relative to the top SM
            # directory, not the song folder.
            candidate = root / asset
        try:
            collapsed = candidate.resolve(strict=False)
            collapsed.relative_to(root.resolve(strict=False))
        except ValueError:
            return None
        return collapsed

    @staticmethod
    def _simfile_root_for_chart(chart_file):
        chart = Path(chart_file).resolve(strict=False)
        for parent in chart.parents:
            if parent.name == 'Songs':
                return parent.parent
        # External AdditionalSongFolders do not expose Etterna's mount root
        # here; the song folder's parent is the closest equivalent.
        song_dir = Path(chart_file).parent
        return song_dir.parent

    @staticmethod
    def _case_insensitive_existing_path(path):
        path = Path(path)
        if path.exists():
            return path
        if path.is_absolute():
            cur = Path(path.anchor)
            parts = path.parts[1:]
        else:
            cur = Path.cwd()
            parts = path.parts
        for part in parts:
            if part in ('', '.'):
                continue
            if part == '..':
                cur = cur.parent
                continue
            exact = cur / part
            if exact.exists():
                cur = exact
                continue
            if not cur.is_dir():
                return None
            try:
                matches = [
                    child for child in cur.iterdir()
                    if child.name.casefold() == part.casefold()
                ]
            except OSError:
                return None
            if not matches:
                return None
            cur = sorted(matches, key=lambda p: p.name)[0]
        return cur if cur.exists() else None

    @staticmethod
    def _attach_chart_hold_ends(replay, found):
        """Merge chart-derived hold end-rows into the replay's hold list.

        Etterna's .bin only records `(head_row, col)` for each hold; the
        tail row lives in the .sm/.ssc. Idempotent — a second call with
        already-3-tuple holds is a no-op."""
        from analysis.games.etterna.sm_chart import parse_notes_block
        chart = (found or {}).get('chart') or {}
        notedata = chart.get('notedata')
        if not notedata:
            return
        replay_holds = replay.get('holds') or []
        if not replay_holds:
            return
        if all(len(h) == 3 for h in replay_holds):
            return
        try:
            _, chart_holds = parse_notes_block(notedata)
        except Exception:
            return
        ends = {(head, col): end for (head, col, end) in chart_holds}
        merged = []
        for h in replay_holds:
            if len(h) == 3:
                merged.append(h)
                continue
            head, col = int(h[0]), int(h[1])
            end = ends.get((head, col))
            merged.append((head, col, end) if end is not None
                          else (head, col))
        replay['holds'] = merged

    @staticmethod
    def _attach_chart_extras(replay, found):
        """Pull chart-only note types (mines, lifts, fakes, roll heads)
        off the .sm/.ssc and stash them on the replay dict.

        These notes never show up in the .bin replay stream: mines and
        fakes aren't judged (PlayerReplay.cpp excludes them), lifts score
        on key release but Etterna's replay writer doesn't emit a row
        for them either, and roll heads are encoded as HoldHead (enum 2)
        in the replay — indistinguishable from regular holds without the
        chart. So the renderer gets them from here or not at all.

        Writes:
          ``chart_mines``  — list of (row, col)
          ``chart_lifts``  — list of (row, col)
          ``chart_fakes``  — list of (row, col)
          ``roll_heads``   — set of (row, col); lets the LN renderer
                             flip holds to roll-colored tails."""
        from analysis.games.etterna.sm_chart import (parse_notes_block,
                                                      NT_MINE, NT_LIFT,
                                                      NT_FAKE, NT_ROLL_HEAD)
        chart = (found or {}).get('chart') or {}
        notedata = chart.get('notedata')
        if not notedata:
            return
        if 'chart_mines' in replay:
            return  # idempotent
        try:
            notes, _ = parse_notes_block(notedata)
        except Exception:
            return
        mines, lifts, fakes, rolls = [], [], [], set()
        for (row, col, nt) in notes:
            if nt == NT_MINE:
                mines.append((row, col))
            elif nt == NT_LIFT:
                lifts.append((row, col))
            elif nt == NT_FAKE:
                fakes.append((row, col))
            elif nt == NT_ROLL_HEAD:
                rolls.add((row, col))
        replay['chart_mines'] = mines
        replay['chart_lifts'] = lifts
        replay['chart_fakes'] = fakes
        replay['roll_heads'] = rolls

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

        def _r2t(row):
            if bpms is not None:
                return row_to_time(int(row), bpms, sm_offset)
            # 120bpm, 48 rows/beat => 96 rows/sec
            return float(row) / 96.0

        if bpms is not None:
            times = np.array([row_to_time(int(r), bpms, sm_offset)
                              for r in replay['noterows']])
        else:
            times = replay['noterows'].astype(np.float64) / 96.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = _r2t(h[2])

        # Chart-derived mines/lifts/fakes — converted to time-space once
        # here so the renderer can read prebuilt arrays instead of
        # redoing row→time per frame. Absent when resolve_all never ran
        # (e.g. the chart file couldn't be found); renderer tolerates
        # missing keys.
        for src_key, t_key, c_key in (
                ('chart_mines', 'mine_times', 'mine_cols'),
                ('chart_lifts', 'lift_times', 'lift_cols'),
                ('chart_fakes', 'fake_times', 'fake_cols')):
            src = replay.get(src_key)
            if not src:
                continue
            ts = np.array([_r2t(r) for (r, _c) in src], dtype=np.float64)
            cs = np.array([c for (_r, c) in src], dtype=np.int32)
            order = np.argsort(ts, kind='stable')
            replay[t_key] = ts[order]
            replay[c_key] = cs[order]

        return times, hold_tails, infer_keycount(replay)

    def judge_label(self, replay, judge=None, **_):
        return str(judge or 'J4')

    def default_scroll_mode(self):
        return 'cmod'

    def player_kwargs(self, replay, judge=None, **_):
        return {'ett_judge': judge or 'J4'}

    # --- library scan -----------------------------------------------------
    _STEPSTYPE_KEYCOUNT = {
        'dance-single': 4, 'dance-solo': 6, 'dance-double': 8,
        'pump-single': 5, 'pump-double': 10, 'kb7-single': 7,
    }

    def scan_library(self, progress=None):
        from analysis.games.etterna.replay import (parse_etterna_xml,
                                                   find_etterna_dirs)
        dirs = find_etterna_dirs()
        xml = dirs.get('xml_path')
        replays = dirs.get('replays_dir')
        if not xml or not replays:
            return []
        out = []
        rdir = Path(replays)
        for s in parse_etterna_xml(xml):
            rp = rdir / s['scorekey']
            if not rp.exists():
                continue
            out.append({
                'game': 'etterna',
                'replay_path': str(rp),
                'scorekey': s['scorekey'],
                'song': s.get('song', ''),
                'pack': s.get('pack', ''),
                'steps': s.get('steps', ''),
                'rate': s.get('rate', 1.0),
                'wife': s.get('ssrnormpercent', 0),
                'grade': s.get('grade', ''),
                'datetime': s.get('datetime', ''),
                'mtime': rp.stat().st_mtime,
                'ssrs': s.get('ssrs', {}),
                'maxcombo': s.get('maxcombo', 0),
                'chart_key': s.get('chartkey', ''),
                'keycount': self._STEPSTYPE_KEYCOUNT.get(
                    s.get('stepstype', 'dance-single'), 4),
                'judgescale': float(s.get('judgescale', 1.0)),
                # Etterna.xml's TapNoteScores block — includes HitMine /
                # AvoidMine alongside the tap counts. The replay .bin
                # doesn't record which mines were hit, so this is the only
                # way to surface mine-hit info in the player.
                'judgments': dict(s.get('judgments') or {}),
            })
        return out

    # --- standalone-launch resolver --------------------------------------
    def can_handle_path(self, path):
        # Etterna claims anything that isn't clearly another game's replay.
        # osu's can_handle_path matches first on .osr; this is the fallback.
        return not str(path).lower().endswith('.osr')

    def resolve_standalone(self, path, args=None):
        from analysis.games.etterna.replay import parse_replay
        args = args or []
        rep = parse_replay(path)
        bpms = None
        sm_off = 0.0
        audio = None
        if '--bpm' in args:
            bpms = [(0.0, float(args[args.index('--bpm') + 1]))]
        if '--sm' in args:
            from analysis.games.etterna.sm_chart import parse_sm, parse_ssc
            smp = args[args.index('--sm') + 1]
            data = parse_ssc(smp) if smp.endswith('.ssc') else parse_sm(smp)
            bpms = data['bpms']
            sm_off = data['offset']
            audio = EtternaAdapter._resolve_music_asset(smp, data['music'])
        if '--audio' in args:
            audio = args[args.index('--audio') + 1]
        return rep, bpms, sm_off, audio, {}

    # --- PlayerTab kwargs -------------------------------------------------
    def player_tab_kwargs(self, replay, entry, chart_ctx):
        bpms, sm_off, _audio = chart_ctx
        return {
            'bpms': bpms,
            'sm_offset': sm_off,
            'xml_judgments': entry.get('judgments'),
        }

    # --- note visualizer --------------------------------------------------
    def viz_windows(self, replay, judge=None, od=None):
        from analysis.viz.note_visualizer import etterna_windows
        j = judge or 'J4'
        return etterna_windows(j), 'noterow', 0.37


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
