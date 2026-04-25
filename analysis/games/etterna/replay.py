"""Etterna ReplaysV2 parser and Etterna.xml scores metadata parser."""
import os
import sys
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


MISS_SENTINEL = 1.000000
TAP_NOTE_TYPE_HOLD_HEAD = 2
TAP_NOTE_TYPE_MINE = 4


def _sanitize_nonascii(raw: bytes) -> bytes:
    """Rewrite *invalid-UTF-8* bytes as `&#NN;` numeric entities while
    leaving valid UTF-8 sequences untouched. Etterna.xml is declared
    UTF-8 and is mostly valid UTF-8 (Japanese/Greek song names work
    fine), but occasionally contains a stray latin-1 byte (e.g.
    `Punkt\xfcre` ; `ü` as 0xFC) that breaks the parser. Escaping every
    high byte would corrupt the legit multi-byte characters, so we
    round-trip through `utf-8` with `backslashreplace` to isolate just
    the bad bytes and rewrite those as entities."""
    try:
        raw.decode('utf-8')
        return raw  # fully valid UTF-8; no rewriting needed
    except UnicodeDecodeError:
        pass
    text = raw.decode('utf-8', errors='backslashreplace')
    # `backslashreplace` emits `\xNN` for each invalid byte. Swap those
    # for `&#NN;` so the XML parser accepts them verbatim.
    pattern = re.compile(r'\\x([0-9a-fA-F]{2})')
    fixed = pattern.sub(lambda m: f'&#{int(m.group(1), 16)};', text)
    return fixed.encode('utf-8')


def _strip_xml_decl(raw: bytes) -> bytes:
    """Drop a leading ``<?xml ... ?>`` declaration. Useful when the
    declaration lies about the encoding; without it, the parser treats
    the bytes as ASCII/UTF-8 and our entity rewrite takes over."""
    if raw.startswith(b'<?xml'):
        end = raw.find(b'?>')
        if end != -1:
            return raw[end + 2:].lstrip()
    return raw


def parse_replay(filepath):
    """Parse an Etterna .bin replay file. Handles V1 and V2 interchangeably.

    Format (per Replay.cpp in etternagame/etterna):

    V2 ; one line per replay event:
        <noterow> <offset> <track> [<TapNoteType>]
            The 4th field is only written when it's not TapNoteType_Tap.
            Etterna's enum uses HoldHead=2 and Mine=4. Mine-hit events
            are useful input data, but they are not tap notes and should
            not be fed into the main note renderer/timing stats.

        H <noterow> <track> [<HoldNoteScore>]
            One line per *dropped* hold (HoldReplayResult). These are
            NOT hold-head declarations ; every hold gets its head
            encoded via TapNoteType above. The dropped-hold list is the
            v2 equivalent of "player let go mid-hold".

    V1 ; the legacy basic format:
        <noterow> <offset>
            Only two tokens per line; no track column, no note type,
            no H lines. Detected by `len(parts) < 3`. Etterna falls
            back to LoadReplayDataBasic() for these files. Column is
            unknown so we can't draw per-lane judgments, but timing
            stats (histogram/mean/stdev) still work.

    Hold-head tails aren't in the replay at all; they come from the
    chart and are joined in later (see EtternaAdapter.resolve_all)."""
    noterows, offsets, columns, notetypes = [], [], [], []
    dropped_holds = []
    is_v1 = False
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('H '):
                parts = line.split()
                if len(parts) >= 3:
                    dropped_holds.append((int(parts[1]), int(parts[2])))
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                nr = int(parts[0])
                off = float(parts[1])
            except ValueError:
                continue
            if len(parts) < 3:
                # V1 basic format: no column info. Use sentinel -1 so the
                # downstream filters can still compute per-row stats; the
                # lane-aware renderer will skip these.
                is_v1 = True
                col = -1
                nt = 0
            else:
                try:
                    col = int(parts[2])
                except ValueError:
                    continue
                nt = int(parts[3]) if len(parts) >= 4 else 0
            noterows.append(nr)
            offsets.append(off)
            columns.append(col)
            notetypes.append(nt)

    noterows = np.array(noterows, dtype=np.int64)
    offsets = np.array(offsets, dtype=np.float64)
    columns = np.array(columns, dtype=np.int32)
    notetypes = np.array(notetypes, dtype=np.int32)
    mine_mask = notetypes == TAP_NOTE_TYPE_MINE
    mine_hits = [(int(noterows[i]), int(columns[i]), float(offsets[i]))
                 for i in np.flatnonzero(mine_mask)]
    if np.any(mine_mask):
        keep = ~mine_mask
        noterows = noterows[keep]
        offsets = offsets[keep]
        columns = columns[keep]
        notetypes = notetypes[keep]

    misses = np.isclose(offsets, MISS_SENTINEL)

    # TapNoteType_HoldHead == 2 ; one entry per hold head actually judged.
    holds = [(int(noterows[i]), int(columns[i]))
             for i in np.flatnonzero(notetypes == TAP_NOTE_TYPE_HOLD_HEAD)]

    return {
        'noterows': noterows,
        'offsets': offsets,
        'columns': columns,
        'notetypes': notetypes,
        'misses': misses,
        'holds': holds,
        'dropped_holds': dropped_holds,
        'mine_hits': mine_hits,
        'replay_version': 1 if is_v1 else 2,
        'filepath': str(filepath),
    }


def clean_offsets(replay):
    """Return offsets & columns with misses removed (for timing analysis)."""
    m = ~replay['misses']
    return {
        'noterows': replay['noterows'][m],
        'offsets': replay['offsets'][m],
        'columns': replay['columns'][m],
        'notetypes': replay['notetypes'][m],
    }


def parse_etterna_xml(filepath):
    """Return a list of score metadata dicts."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        # Some Etterna.xml files have malformed entries. Strip control
        # bytes and rewrite stray non-ASCII bytes (e.g. `Punkt\xfcre` ;
        # latin-1 `ü` in a file declared UTF-8) as numeric entities so
        # the stdlib parser accepts them. Also retries with the XML
        # declaration stripped in case the encoding claim itself is
        # wrong.
        with open(filepath, 'rb') as f:
            raw = f.read()
        import re as _re
        raw = _re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw)
        root = None
        for attempt in (raw, _sanitize_nonascii(raw),
                        _strip_xml_decl(_sanitize_nonascii(raw))):
            try:
                root = ET.fromstring(attempt)
                break
            except ET.ParseError:
                continue
    ps = root.find('PlayerScores')
    if ps is None:
        return []
    scores = []
    for chart in ps.findall('Chart'):
        chartkey = chart.get('Key', '')
        pack = chart.get('Pack', '')
        song = chart.get('Song', '')
        steps = chart.get('Steps', '')
        stepstype = chart.get('StepsType', 'dance-single')
        for sa in chart.findall('ScoresAt'):
            rate = float(sa.get('Rate', '1.0'))
            pbkey = sa.get('PBKey', '')
            for score in sa.findall('Score'):
                scorekey = score.get('Key', '')
                meta = {
                    'scorekey': scorekey,
                    'chartkey': chartkey,
                    'pack': pack,
                    'song': song,
                    'steps': steps,
                    'stepstype': stepstype,
                    'rate': rate,
                    'pbkey': pbkey,
                    'is_pb': scorekey == pbkey,
                }
                for tag in ('Grade', 'WifeScore', 'SSRNormPercent',
                            'JudgeScale', 'MaxCombo', 'MusicRate',
                            'DateTime', 'TopScore', 'PlayedSeconds',
                            'Modifiers', 'EtternaValid'):
                    el = score.find(tag)
                    if el is not None and el.text is not None:
                        meta[tag.lower()] = el.text

                for numtag in ('WifeScore', 'SSRNormPercent', 'JudgeScale',
                               'MaxCombo', 'PlayedSeconds', 'TopScore'):
                    k = numtag.lower()
                    if k in meta:
                        try:
                            meta[k] = float(meta[k])
                        except ValueError:
                            pass

                ssr = score.find('SkillsetSSRs')
                if ssr is not None:
                    meta['ssrs'] = {c.tag: float(c.text) for c in ssr
                                    if c.text is not None}

                tns = score.find('TapNoteScores')
                if tns is not None:
                    meta['judgments'] = {c.tag: int(c.text) for c in tns
                                         if c.text is not None and c.text.strip().lstrip('-').isdigit()}

                scores.append(meta)
    return scores


def find_replay_for_score(scorekey, replays_dir):
    if not scorekey:
        return None
    p = Path(replays_dir) / scorekey
    if p.exists():
        return str(p)
    return None


def _etterna_root_override():
    """Read a user-set install-root override from the GUI settings layer, if
    any. Done via lazy import so the core module stays importable without Qt."""
    try:
        from analysis.gui.settings import get_etterna_root_override
    except Exception:
        return None
    try:
        return get_etterna_root_override()
    except Exception:
        return None


def _parse_additional_song_folders(save):
    """Read AdditionalSongFolders / AdditionalFolders from Preferences.ini.
    Etterna writes these semicolon-separated; we also accept commas defensively.
    Returns a list of absolute path strings ; only dirs that actually exist."""
    prefs = Path(save) / 'Preferences.ini'
    if not prefs.is_file():
        return []
    out = []
    try:
        for line in prefs.read_text(errors='ignore').splitlines():
            s = line.strip()
            if not s or '=' not in s:
                continue
            key, _, val = s.partition('=')
            if key.strip() not in ('AdditionalSongFolders', 'AdditionalFolders'):
                continue
            for part in val.replace(',', ';').split(';'):
                p = part.strip()
                if p and Path(p).is_dir():
                    out.append(str(Path(p)))
    except OSError:
        return []
    return out


def _resolve_etterna_save(save):
    """Build the dirs dict for an existing save directory. Returns None if
    missing. `save` must point at the Save folder itself (not the install
    root) ; callers handle install-root → Save resolution."""
    save = Path(save)
    if not save.exists():
        return None
    replays = save / 'ReplaysV2'
    xmls: list[Path] = []
    profiles = save / 'LocalProfiles'
    if profiles.exists():
        for sub in sorted(profiles.iterdir()):
            candidate = sub / 'Etterna.xml'
            if candidate.exists():
                xmls.append(candidate)
    direct = save / 'Etterna.xml'
    if not xmls and direct.exists():
        xmls.append(direct)
    # `xml_path` stays as the first profile's XML for backward compat
    # with single-profile callers (batch.py, tests). Multi-profile
    # consumers (the adapter) iterate `xml_paths` to cover every
    # profile under LocalProfiles.
    return {
        'save_dir': str(save),
        'replays_dir': str(replays) if replays.exists() else None,
        'xml_path': str(xmls[0]) if xmls else None,
        'xml_paths': [str(p) for p in xmls],
        'extra_songs_dirs': _parse_additional_song_folders(save),
    }


def _resolve_etterna_root(root):
    """Accept either an install root (contains Save/) or a bare Save dir.
    Returns the resolved dirs dict or None."""
    root = Path(root)
    if not root.exists():
        return None
    save = root / 'Save'
    if save.is_dir() and ((save / 'LocalProfiles').is_dir()
                          or (save / 'Etterna.xml').is_file()):
        return _resolve_etterna_save(save)
    # Back-compat: user pointed directly at Save/.
    return _resolve_etterna_save(root)


def find_etterna_dirs():
    """Returns dict with save_dir, replays_dir, xml_path, extra_songs_dirs.
    User override from GUI settings wins over autodetection."""
    override = _etterna_root_override()
    if override:
        resolved = _resolve_etterna_root(override)
        if resolved is not None:
            return resolved
    candidates = [
        Path.home() / '.etterna',
        Path.home() / 'etterna',
        Path.home() / '.stepmania-5.0',
        Path.home() / '.stepmania-5.1',
    ]
    for root in candidates:
        resolved = _resolve_etterna_root(root)
        if resolved is not None:
            return resolved
    return {'save_dir': None, 'replays_dir': None, 'xml_path': None,
            'xml_paths': [], 'extra_songs_dirs': []}


def summary(replay):
    clean = clean_offsets(replay)
    offs = clean['offsets']
    cols = clean['columns']
    miss_count = int(replay['misses'].sum())
    total = len(replay['offsets'])
    hits = len(offs)
    out = {
        'total_notes': total,
        'hits': hits,
        'misses': miss_count,
        'mean_offset_ms': float(np.mean(offs)) * 1000 if hits else 0,
        'median_offset_ms': float(np.median(offs)) * 1000 if hits else 0,
        'std_offset_ms': float(np.std(offs)) * 1000 if hits else 0,
        'abs_mean_ms': float(np.mean(np.abs(offs))) * 1000 if hits else 0,
    }
    for col in range(4):
        m = cols == col
        if m.any():
            out[f'col{col}_mean_ms'] = float(np.mean(offs[m])) * 1000
            out[f'col{col}_std_ms'] = float(np.std(offs[m])) * 1000
            out[f'col{col}_n'] = int(m.sum())
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        dirs = find_etterna_dirs()
        print("Etterna dirs:")
        for k, v in dirs.items():
            print(f"  {k}: {v}")
        sys.exit(0)
    replay = parse_replay(sys.argv[1])
    s = summary(replay)
    print(f"File: {replay['filepath']}")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
