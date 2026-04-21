"""Etterna ReplaysV2 parser and Etterna.xml scores metadata parser."""
import os
import sys
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


MISS_SENTINEL = 1.000000


def parse_replay(filepath):
    noterows, offsets, columns, notetypes = [], [], [], []
    holds = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('H '):
                parts = line.split()
                if len(parts) >= 3:
                    holds.append((int(parts[1]), int(parts[2])))
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                nr = int(parts[0])
                off = float(parts[1])
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
    misses = np.isclose(offsets, MISS_SENTINEL)

    return {
        'noterows': noterows,
        'offsets': offsets,
        'columns': columns,
        'notetypes': notetypes,
        'misses': misses,
        'holds': holds,
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
        # Some Etterna.xml files have malformed entries. Strip bad bytes and retry.
        with open(filepath, 'rb') as f:
            raw = f.read()
        import re as _re
        # drop control chars except \t, \n, \r
        raw = _re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw)
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            # last resort: lxml with recover=True
            try:
                from lxml import etree as LET
                parser = LET.XMLParser(recover=True, huge_tree=True)
                root = LET.fromstring(raw, parser=parser)
            except Exception as e:
                print(f"Etterna.xml unparseable: {e}")
                return []
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


def _etterna_save_override():
    """Read a user-set save-dir override from the GUI settings layer, if any.
    Done via lazy import so the core module stays importable without Qt."""
    try:
        from analysis.gui.settings import get_etterna_save_override
    except Exception:
        return None
    try:
        return get_etterna_save_override()
    except Exception:
        return None


def _resolve_etterna_save(save):
    """Build the (save_dir, replays_dir, xml_path) dict for an existing save
    directory. Returns None if the dir is missing."""
    save = Path(save)
    if not save.exists():
        return None
    replays = save / 'ReplaysV2'
    xml = None
    profiles = save / 'LocalProfiles'
    if profiles.exists():
        for sub in sorted(profiles.iterdir()):
            candidate = sub / 'Etterna.xml'
            if candidate.exists():
                xml = candidate
                break
    direct = save / 'Etterna.xml'
    if xml is None and direct.exists():
        xml = direct
    return {
        'save_dir': str(save),
        'replays_dir': str(replays) if replays.exists() else None,
        'xml_path': str(xml) if xml else None,
    }


def find_etterna_dirs():
    """Returns dict with save_dir, replays_dir, xml_path. User override from
    GUI settings wins over autodetection."""
    override = _etterna_save_override()
    if override:
        resolved = _resolve_etterna_save(override)
        if resolved is not None:
            return resolved
    candidates = [
        Path.home() / '.etterna' / 'Save',
        Path.home() / '.stepmania-5.0' / 'Save',
        Path.home() / '.stepmania-5.1' / 'Save',
    ]
    for save in candidates:
        resolved = _resolve_etterna_save(save)
        if resolved is not None:
            return resolved
    return {'save_dir': None, 'replays_dir': None, 'xml_path': None}


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
