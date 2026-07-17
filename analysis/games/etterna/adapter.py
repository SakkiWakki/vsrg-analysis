"""Etterna game adapter + scroll modes (CMOD + XMOD)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from analysis.core.cache import Cache
from analysis.core.game import GameAdapter
from analysis.player import scroll


_LIBRARY_CACHE = Cache('etterna_library.pkl')


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
                    return self._remember_song(replay, hit)
            except Exception:
                pass
        try:
            if progress:
                progress('fingerprint chart match…')
            hit = find_chart_for_replay(replay['noterows'],
                                        replay['columns'], songs)
            return self._remember_song(replay, hit)
        except Exception:
            return None

    @staticmethod
    def _remember_song(replay, found):
        """Stash the matched simfile on the replay so late lookups
        (background_path runs at player init, with only the replay in
        hand) don't repeat the chartkey/fingerprint search."""
        if found:
            replay['_song_file'] = found['file']
            replay['_song_background'] = found['data'].get('background', '')
        return found

    def resolve_audio(self, replay, entry=None, progress=None):
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None
        music = found['data'].get('music', '')
        return self._resolve_music_asset(found['file'], music)

    def background_path(self, replay) -> str | None:
        """Resolve #BACKGROUND like Etterna: the tag path first, then
        Song::TidyUpData's scan for an image named like a background."""
        chart_file = replay.get('_song_file')
        if not chart_file:
            return None
        resolved = self._resolve_song_asset(
            chart_file, replay.get('_song_background', ''))
        if resolved:
            return resolved
        for p in self._song_dir_files(chart_file):
            name = p.name.casefold()
            is_image = p.suffix[1:].casefold() in {'png', 'jpg', 'jpeg',
                                                   'bmp', 'gif'}
            if is_image and 'banner' not in name and (
                    'bg' in name or 'background' in name):
                return str(p)
        return None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        found = self._find_chart(replay, entry=entry, progress=progress)
        if not found:
            return None, 0.0
        return found['data']['bpms'], found['data']['offset']

    def resolve_all(self, replay, entry=None, progress=None):
        """Single-pass combined resolver ; avoids parsing the .sm/.ssc twice.
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
        for p in EtternaAdapter._song_dir_files(chart_file):
            if p.suffix[1:].casefold() in {'mp3', 'oga', 'ogg', 'wav'}:
                return str(p)
        return None

    @staticmethod
    def _song_dir_files(chart_file):
        """Files in the simfile's folder, in Etterna's scan order
        (case-insensitive name sort). Empty on I/O failure."""
        song_dir = Path(chart_file).parent
        try:
            return sorted(
                (p for p in song_dir.iterdir() if p.is_file()),
                key=lambda p: p.name.casefold(),
            )
        except OSError:
            return []

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
        # On case-insensitive filesystems (default NTFS, HFS+), exists()
        # returns True even when the input's case differs from the real
        # on-disk name and pathlib won't rewrite the case for us. So
        # skip both exists() fast paths there and always walk with
        # iterdir+casefold to recover the true on-disk spelling, which
        # is what Etterna's FilenameDB promises callers.
        case_sensitive_fs = sys.platform not in ('win32', 'darwin')

        path = Path(path)
        if case_sensitive_fs and path.exists():
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
            if case_sensitive_fs:
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
        tail row lives in the .sm/.ssc. Idempotent ; a second call with
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
        in the replay ; indistinguishable from regular holds without the
        chart. So the renderer gets them from here or not at all.

        Writes:
          ``chart_mines``  ; list of (row, col)
          ``chart_lifts``  ; list of (row, col)
          ``chart_fakes``  ; list of (row, col)
          ``roll_heads``   ; set of (row, col); lets the LN renderer
                             flip holds to roll-colored tails.
          ``keycount``     ; track count from the chart's stepstype."""
        from analysis.games.etterna.sm_chart import (parse_notes_block,
                                                      stepstype_keycount,
                                                      row_to_time,
                                                      NT_TAP, NT_HOLD_HEAD,
                                                      NT_MINE, NT_LIFT,
                                                      NT_FAKE, NT_ROLL_HEAD)
        data = (found or {}).get('data') or {}
        chart = (found or {}).get('chart') or {}
        notedata = chart.get('notedata')
        stepstype = chart.get('stepstype', '')
        if stepstype and 'keycount' not in replay:
            replay['keycount'] = stepstype_keycount(stepstype)
        # Pull raw timing data off the matched chart and project it into
        # the canonical SV doc. The renderer reads `replay['sv']` only ;
        # `prepare_replay_times` and the chart-extras logic below also
        # consume STOPS/DELAYS/WARPS via the same doc when needed.
        scrolls = chart.get('scrolls') or []
        speeds = chart.get('speeds') or []
        stops = chart.get('stops') or []
        delays = chart.get('delays') or []
        warps = chart.get('warps') or []
        bpms = chart.get('bpms') or data.get('bpms') or []
        sm_offset = chart.get('offset') or data.get('offset') or 0.0
        from analysis.player.sv.replay_doc import (SvReplayDoc,
                                                    KIND_BEAT_SPACE,
                                                    KIND_IDENTITY)
        has_sv = bool(scrolls or speeds or len(bpms) > 1
                      or stops or delays or warps)
        if has_sv:
            replay['sv'] = SvReplayDoc(
                engine_kind=KIND_BEAT_SPACE,
                engine_key='etterna_beat',
                scrolls=list(scrolls),
                speeds=list(speeds),
                stops=list(stops),
                delays=list(delays),
                warps=list(warps),
                bpms=list(bpms),
                sm_offset=float(sm_offset),
            )
        else:
            replay['sv'] = SvReplayDoc(
                engine_kind=KIND_IDENTITY, engine_key='identity')
        if not notedata:
            return
        if 'chart_mines' in replay:
            return  # idempotent
        try:
            notes, _ = parse_notes_block(notedata)
        except Exception:
            return
        chart_warps = chart.get('warps') or []
        chart_stops = chart.get('stops') or []
        chart_delays = chart.get('delays') or []
        from analysis.games.etterna.sm_chart import is_beat_in_warp

        def _in_warp(row):
            return is_beat_in_warp(row / 48.0, chart_warps,
                                   chart_stops, chart_delays)

        mines, lifts, fakes, rolls = [], [], [], set()
        chart_max_row = 0
        for (row, col, nt) in notes:
            if nt == NT_MINE:
                mines.append((row, col))
            elif nt == NT_LIFT:
                lifts.append((row, col))
            elif nt == NT_FAKE:
                fakes.append((row, col))
            elif nt == NT_ROLL_HEAD:
                rolls.add((row, col))
            if nt in (NT_TAP, NT_HOLD_HEAD, NT_ROLL_HEAD) and _in_warp(row):
                # Old .sm negative-BPM/negative-stop warps can contain arrows
                # that Etterna's replay stream never judges. Keep them visible
                # as fake notes for analysis (and beauty)
                fakes.append((row, col))
            if row > chart_max_row:
                chart_max_row = row
        replay['chart_mines'] = mines
        replay['chart_lifts'] = lifts
        replay['chart_fakes'] = fakes
        replay['roll_heads'] = rolls

        # Inject post-death chart notes as misses so prepare_replay_times,
        # note culling, t_max, and the standard miss renderer handle them
        # without a separate code path.
        import numpy as np
        replay_max_row = int(replay['noterows'].max()) if len(replay['noterows']) else 0
        _, chart_holds = parse_notes_block(notedata)
        hold_ends = {(hr, hc): er for (hr, hc, er) in chart_holds}

        missing = []  # (row, col, is_hold)
        for (row, col, nt) in notes:
            if row <= replay_max_row:
                continue
            if nt not in (NT_TAP, NT_HOLD_HEAD, NT_ROLL_HEAD):
                continue
            missing.append((row, col, nt in (NT_HOLD_HEAD, NT_ROLL_HEAD)))

        if not missing:
            return

        bpms = chart.get('bpms') or data.get('bpms') or []
        offset = chart.get('offset') or data.get('offset') or 0.0
        stops = chart.get('stops') or []
        delays = chart.get('delays') or []
        warps = chart.get('warps') or []

        if chart_max_row - replay_max_row > 192:
            replay['death_time'] = row_to_time(replay_max_row, bpms, offset,
                                                stops, delays, warps)

        new_rows = np.array([r for r, _c, _h in missing], dtype=np.int64)
        new_cols = np.array([c for _r, c, _h in missing], dtype=np.int32)
        new_offs = np.full(len(missing), 1.0, dtype=np.float64)  # MISS_SENTINEL
        new_nts  = np.zeros(len(missing), dtype=np.int32)

        rows = np.concatenate([replay['noterows'], new_rows])
        order = np.argsort(rows, kind='stable')
        replay['noterows']  = rows[order]
        replay['columns']   = np.concatenate([replay['columns'],  new_cols])[order]
        replay['offsets']   = np.concatenate([replay['offsets'],  new_offs])[order]
        replay['notetypes'] = np.concatenate([replay['notetypes'],new_nts])[order]
        replay['misses']    = np.concatenate([replay['misses'],
                                              np.ones(len(missing), dtype=bool)])[order]

        existing_holds = replay.get('holds') or []
        new_holds = [(r, c, hold_ends[(r, c)])
                     for r, c, is_hold in missing
                     if is_hold and (r, c) in hold_ends]
        replay['holds'] = existing_holds + new_holds

    def judgement_windows(self, replay, judge=None, **_):
        from analysis.games.etterna.judgment import windows_for
        return windows_for(judge or 'J4')

    def nudge_judge(self, current, delta):
        """Step through J1..J9 by one per click. Etterna's judge is
        discrete ; integers only ; so we take the sign of `delta` and
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

    def prepare_replay_times(self, replay, bpms=None, sm_offset=0.0,
                             keycount=None, **_):
        import numpy as np
        from analysis.games.etterna.sm_chart import row_to_time

        # Chart-resolved STOPS/DELAYS/WARPS ; these affect real time via
        # pauses and beat-space teleports, so row->time has to use them for
        # anything better than constant-BPM charts. Absent when the chart
        # didn't resolve; in that case row_to_time falls back to BPM-only.
        from analysis.player.sv.replay_doc import replay_sv
        sv_doc = replay_sv(replay)
        stops = list(sv_doc.stops)
        delays = list(sv_doc.delays)
        warps = list(sv_doc.warps)

        def _r2t(row):
            if bpms is not None:
                return row_to_time(int(row), bpms, sm_offset,
                                    stops, delays, warps)
            # 120bpm, 48 rows/beat => 96 rows/sec
            return float(row) / 96.0

        def _active_until(row):
            beat = int(row) / 48.0
            for wb, wl in warps:
                if wb <= beat < wb + wl:
                    return row_to_time(int(round(wb * 48)), bpms, sm_offset,
                                       stops, delays, warps)
            return float('inf')

        if bpms is not None:
            times = np.array([row_to_time(int(r), bpms, sm_offset,
                                           stops, delays, warps)
                              for r in replay['noterows']])
        else:
            times = replay['noterows'].astype(np.float64) / 96.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = _r2t(h[2])

        # Chart-derived mines/lifts/fakes; converted to time-space once
        # here so the renderer can read prebuilt arrays instead of
        # redoing row→time per frame
        for src_key, t_key, c_key in (
                ('chart_mines', 'mine_times', 'mine_cols'),
                ('chart_lifts', 'lift_times', 'lift_cols'),
                ('chart_fakes', 'fake_times', 'fake_cols')):
            src = replay.get(src_key)
            if not src:
                continue
            ts = np.array([_r2t(r) for (r, _c) in src], dtype=np.float64)
            rs = np.array([r for (r, _c) in src], dtype=np.int64)
            until = np.array([_active_until(r) for (r, _c) in src],
                             dtype=np.float64)
            cs = np.array([c for (_r, c) in src], dtype=np.int32)
            order = np.argsort(ts, kind='stable')
            replay[t_key] = ts[order]
            replay[t_key.replace('_times', '_rows')] = rs[order]
            replay[t_key.replace('_times', '_until')] = until[order]
            replay[c_key] = cs[order]

        kc = keycount or replay.get('keycount') or 4
        return times, hold_tails, int(kc)

    def judge_label(self, replay, judge=None, **_):
        return str(judge or 'J4')

    def default_scroll_mode(self):
        return 'cmod'

    def player_kwargs(self, replay, judge=None, **_):
        return {'ett_judge': judge or 'J4'}

    # ── Cross-game mod hooks ──

    def mods_short(self, replay) -> str:
        rate = self._etterna_rate(replay)
        return '' if abs(rate - 1.0) < 1e-4 else f'{rate:g}x'

    def mods_raw(self, replay) -> dict:
        return {'rate': self._etterna_rate(replay), 'flags': []}

    def mods_rate_multiplier(self, replay) -> float:
        return self._etterna_rate(replay)

    def chart_stats_extra(self, replay):
        cm = (replay or {}).get('chart_meta') or {}
        msd = float(cm.get('msd', 0) or 0)   # overall MSD
        extra = {k: float(v) for k, v in (cm.get('msd_skills') or {}).items()}
        return msd, msd, extra

    @staticmethod
    def _etterna_rate(replay) -> float:
        if not isinstance(replay, dict):
            return 1.0
        r = replay.get('rate') or (replay.get('meta') or {}).get('rate')
        try:
            return float(r) if r is not None else 1.0
        except (TypeError, ValueError):
            return 1.0

    # --- modfiles (#FGCHANGES .lua) ---------------------------------------
    # Etterna HAS replays, so modfile compilation ENHANCES existing replay
    # playback: the SM5 actor-tree Lua runs once under a stub environment
    # (sm5_env / modfile) to harvest per-note mod events (poptions method
    # calls) and storyboard actors. Shares one memoized compile between
    # note_mods / storyboard so the Lua runs at most once per replay.

    def _modfile_simfile(self, replay):
        """The .sm/.ssc backing this replay, resolved once _find_chart has
        stashed it (`resolve_all`/`background_path` run before the effect
        build). None when the chart was never matched."""
        return (replay or {}).get('_song_file')

    def _compiled_modfile(self, replay):
        cached = replay.get('_etterna_modfile')
        if cached is not None:
            return cached or None
        sm_path = self._modfile_simfile(replay)
        if not sm_path:
            return None
        from analysis.games.etterna.modfile import compile_modfile
        compiled = compile_modfile(sm_path)
        replay['_etterna_modfile'] = compiled or {}
        return compiled

    def note_mods(self, replay):
        from analysis.games.etterna.modfile import compile_mod_channels
        from analysis.games.etterna.sm_chart import parse_sm, parse_ssc
        from analysis.games.notitg.note_mods import NotitgNoteMods
        compiled = self._compiled_modfile(replay)
        if not compiled or not compiled.get('mod_events'):
            return None
        channels = compile_mod_channels(compiled['mod_events'])
        sm_path = self._modfile_simfile(replay)
        data = (parse_ssc(sm_path) if str(sm_path).endswith('.ssc')
                else parse_sm(sm_path))
        return NotitgNoteMods(channels, data['bpms'])

    def storyboard(self, replay):
        """Modfile actors (Def.Quad/Sprite/BitmapText prank overlays and
        ActorFrame groups) render through the storyboard pipeline in SM's
        640x480 screen space. The nested `tree` (ActorFrame = a group
        whose transform composes onto children) is preferred; the flat
        `elements` list is the fallback for charts with no hierarchy."""
        from analysis.player.render.storyboard import Storyboard
        compiled = self._compiled_modfile(replay) or {}
        elements = compiled.get('tree') or compiled.get('elements')
        if not elements:
            return None
        return Storyboard(design_w=640.0, design_h=480.0, fit='height',
                          elements=tuple(elements))

    # --- library scan -----------------------------------------------------
    _STEPSTYPE_KEYCOUNT = {
        'dance-single': 4, 'dance-solo': 6, 'dance-double': 8,
        'pump-single': 5, 'pump-double': 10, 'kb7-single': 7,
    }
    # Why not just do this in the first place wtf
    def scan_library(self, progress=None):
        from analysis.games.etterna.replay import find_etterna_dirs
        dirs = find_etterna_dirs()
        xmls = self._xml_paths(dirs)
        replays = dirs.get('replays_dir')
        if not xmls or not replays:
            return []
        return self._entries_from_xmls(xmls, Path(replays),
                                        ck2st=self._load_chartkey_stepstype(dirs))

    @staticmethod
    def _xml_paths(dirs):
        """Return the list of per-profile Etterna.xml paths. Falls back
        to the singular `xml_path` (old consumers that didn't populate
        `xml_paths`) so external callers building their own dirs dict
        keep working."""
        paths = list(dirs.get('xml_paths') or [])
        if not paths and dirs.get('xml_path'):
            paths = [dirs['xml_path']]
        return paths

    @staticmethod
    def _load_chartkey_stepstype(dirs) -> dict:
        """Build a ``{chartkey: stepstype}`` map from Etterna's
        ``Cache/cache.db`` so scores for non-4K keymodes (kb7, dance-solo,
        pump, etc.) get the right keycount. Etterna.xml's ``<Chart>`` has
        no StepsType attribute, so without this lookup every score ends
        up labeled dance-single by default. Returns ``{}`` if the cache
        is missing or unreadable ; callers fall back to dance-single for
        each unresolved chartkey, matching the old behavior."""
        save = dirs.get('save_dir')
        if not save:
            return {}
        # Cache lives next to Save/ under the install root (one level up).
        cache_db = Path(save).parent / 'Cache' / 'cache.db'
        if not cache_db.is_file():
            return {}
        import sqlite3
        try:
            con = sqlite3.connect(f'file:{cache_db}?mode=ro', uri=True)
            try:
                cur = con.execute('SELECT CHARTKEY, STEPSTYPE FROM steps')
                return {ck: st for ck, st in cur if ck and st}
            finally:
                con.close()
        except sqlite3.DatabaseError as exc:
            print(f'etterna cache.db unreadable: {exc}')
            return {}

    def _score_to_entry(self, s, rdir: Path, ck2st=None):
        rp = rdir / s['scorekey']
        if not rp.exists():
            return None
        # Etterna.xml doesn't carry StepsType on <Chart>; resolve it from
        # Cache/cache.db when available so non-4K keymodes (kb7,
        # dance-solo, pump, etc.) get the right keycount. Falls back to
        # the XML's value (or the dance-single default) when the cache
        # doesn't know the chart.
        stepstype = s.get('stepstype', 'dance-single')
        if ck2st:
            stepstype = ck2st.get(s.get('chartkey', ''), stepstype)
        return {
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
            'stepstype': stepstype,
            'keycount': self._STEPSTYPE_KEYCOUNT.get(stepstype, 4),
            'judgescale': float(s.get('judgescale', 1.0)),
            # Etterna.xml's TapNoteScores block; includes HitMine /
            # AvoidMine alongside the tap counts. The replay .bin
            # doesn't record which mines were hit, so this is the only
            # way to surface mine-hit info in the player.
            'judgments': dict(s.get('judgments') or {}),
        }

    def _entries_from_xml(self, xml_path, rdir: Path, want_keys=None,
                          ck2st=None):
        """Parse a single Etterna.xml and return entries. `want_keys` is
        used by the incremental path to skip already-cached scorekeys;
        `ck2st` is the chartkey→stepstype map from Cache/cache.db so
        non-4K keymodes get the right keycount."""
        from analysis.games.etterna.replay import parse_etterna_xml
        out = []
        for s in parse_etterna_xml(xml_path):
            if want_keys is not None and s['scorekey'] not in want_keys:
                continue
            entry = self._score_to_entry(s, rdir, ck2st=ck2st)
            if entry is not None:
                out.append(entry)
        return out

    def _entries_from_xmls(self, xml_paths, rdir: Path, want_keys=None,
                           ck2st=None):
        """Merge entries from every profile's Etterna.xml. Deduped by
        scorekey ; the replay .bin is shared across profiles via the
        ReplaysV2 folder, so if two profiles reference the same score
        we keep the first occurrence (profiles are iterated in sorted
        order, so 00000000 wins)."""
        out = []
        seen: set[str] = set()
        for xml in xml_paths:
            for entry in self._entries_from_xml(xml, rdir, want_keys,
                                                ck2st=ck2st):
                sk = entry.get('scorekey')
                if sk and sk in seen:
                    continue
                if sk:
                    seen.add(sk)
                out.append(entry)
        return out

    # --- library cache lifecycle -----------------------------------------
    def load_cached(self):
        return _LIBRARY_CACHE.load()

    def save_cached(self, entries):
        _LIBRARY_CACHE.save([e for e in entries if e.get('game') == 'etterna'])

    def rebuild(self, progress=None):
        from analysis.games.etterna.replay import find_etterna_dirs
        _LIBRARY_CACHE.clear()
        dirs = find_etterna_dirs()
        xmls = self._xml_paths(dirs)
        replays = dirs.get('replays_dir')
        if not xmls or not replays:
            return []
        if progress:
            progress(f'etterna: rebuilding from {len(xmls)} profile(s)…')
        entries = self._entries_from_xmls(
            xmls, Path(replays),
            ck2st=self._load_chartkey_stepstype(dirs))
        # Don't poison the cache with an empty result, e.g. if the XML was
        # unparseable
        if entries:
            _LIBRARY_CACHE.save(entries)
        return entries

    def incremental_update(self, progress=None):
        from analysis.games.etterna.replay import find_etterna_dirs
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return self.rebuild(progress=progress)
        dirs = find_etterna_dirs()
        xmls = self._xml_paths(dirs)
        replays = dirs.get('replays_dir')
        if not xmls or not replays:
            return cached

        # Cheap diff: parse every profile's XML just to collect ScoreKeys,
        # then only materialize entries for keys we haven't seen. A
        # ScoreKey is immutable per replay (Etterna mints a new one for
        # every play), so set-difference is exact across profiles too.
        known_keys = {e.get('scorekey') for e in cached if e.get('scorekey')}
        from analysis.games.etterna.replay import parse_etterna_xml
        rdir = Path(replays)
        ck2st = self._load_chartkey_stepstype(dirs)
        new_entries = []
        seen_new: set[str] = set()
        for xml in xmls:
            for s in parse_etterna_xml(xml):
                sk = s['scorekey']
                if sk in known_keys or sk in seen_new:
                    continue
                entry = self._score_to_entry(s, rdir, ck2st=ck2st)
                if entry is not None:
                    new_entries.append(entry)
                    seen_new.add(sk)

        if not new_entries:
            return cached
        if progress:
            progress(f'etterna: {len(new_entries)} new score(s)')
        merged = cached + new_entries
        _LIBRARY_CACHE.save(merged)
        return merged

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
        out = {
            'bpms': bpms,
            'sm_offset': sm_off,
            'xml_judgments': entry.get('judgments'),
            'keycount': entry.get('keycount'),
        }
        out.update(_scroll_kwargs_from_modifiers(entry.get('modifiers')))
        return out

    # --- note visualizer --------------------------------------------------
    def viz_windows(self, replay, judge=None, od=None):
        from analysis.viz.note_visualizer import etterna_windows
        j = judge or 'J4'
        return etterna_windows(j), 'noterow', 0.37

    def populate_notes_model(self, replay, model) -> None:
        _build_chart_extras(model, replay)


def _build_chart_extras(m, replay):
    """Copy Etterna chart-only streams (mines, lifts, fakes, rolls) from
    the replay dict. The adapter populates these during prepare_replay_times
    when a chart match was found; they're absent for osu."""
    from analysis.player.init.notes_model import copy_chart_streams
    copy_chart_streams(m, replay)
    roll_heads = replay.get('roll_heads')
    if roll_heads:
        m.roll_head_keys = set(roll_heads)


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


def _scroll_kwargs_from_modifiers(modifiers):
    """Infer the replay's native Etterna scroll mode from Etterna.xml.

    Old negative-BPM gimmicks are authored for beat-space XMOD. If the GUI
    opens a ``1.23x`` replay in its saved CMOD preference, warp aliases stack
    because CMOD uses seconds-until-note instead of beat distance.
    """
    import re

    text = str(modifiers or '')
    out = {}

    cmod = re.search(r'(?i)(?:^|[\s,])c(?:mod)?\s*([0-9]+(?:\.[0-9]+)?)', text)
    if cmod:
        out['scroll_mode'] = 'cmod'
        out['cmod_bpm'] = float(cmod.group(1))
        return out

    xmod = re.search(r'(?i)(?:^|[\s,])([0-9]+(?:\.[0-9]+)?)\s*x(?:mod)?(?:[\s,]|$)', text)
    if xmod:
        out['scroll_mode'] = 'xmod'
        out['xmod_value'] = float(xmod.group(1))
    return out


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
    scaling here ; the user said 'if something exists for CMOD but not for
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
