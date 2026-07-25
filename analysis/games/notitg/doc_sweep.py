"""Sweep NotITG charts through the element parity harness.

Two charts is not a corpus. This compiles every chart it is pointed at,
builds the static doc, and reports two things per chart:

  - PLACEMENT: the drawn-quad corner error against the legacy painter, in
    design pixels (`drawable_doc.element_parity_report`).
  - COVERAGE: the element kinds and verbs the doc SKIPS. A placement number
    only describes what the doc emitted; it cannot see what was never
    emitted at all, which is how 118 missing rects hid behind a 0.001px
    result on gat 1.

Usage:
    python -m analysis.games.notitg.element_sweep [--limit N] [--root DIR] [--json OUT]

Charts compile in ~10-25s each, so a full library run is hours - start with
a limit and widen once the fast failures are gone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault('VSRG_DRAWABLE_ELEMENTS', '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Sample times as FRACTIONS of the chart's own horizon, not fixed seconds.
# Past the horizon nothing is recorded: the exported channel holds its last
# value while `SegCurve.sample` clamps to the frontier or serves a preview,
# so the two legitimately disagree in territory neither renders. Fixed
# seconds sampled a 117s chart at 300s and reported it 8px off.
#
# The offsets are DELIBERATELY not round fractions. Channel breakpoint times
# are f32, so a segment starting at 60.00000000000378 rounds to exactly 60.0
# and wins the lookup at t=60.0 that the f64 timeline does not. That window is
# ~4e-12s wide - unreachable from an audio clock, but a round sample time
# lands in it every time, and it made a clean chart read as 16px off.
SAMPLE_FRACTIONS = (0.07, 0.13, 0.21, 0.34, 0.47, 0.58, 0.71, 0.83, 0.94)


def sample_times(compiled) -> list:
    """Times to compare at, spread across the chart's recorded span."""
    from analysis.games.notitg.drawable_doc import _horizon

    horizon = _horizon(compiled)
    live = compiled.get('_live_sim')
    frontier = getattr(live, 'frontier', None)
    if isinstance(frontier, (int, float)) and frontier > 0.0:
        horizon = min(horizon, float(frontier))
    return [round(horizon * f, 3) + 0.037 for f in SAMPLE_FRACTIONS]


def _natural_lookup():
    """`element -> (w, h)` logical frame size, read from the asset header the
    way the executor reads it from the uploaded texture."""
    from PySide6.QtGui import QImageReader
    from analysis.player.render.storyboard.asset_size import (
        AssetSizeSpec, resolve)

    cache: dict = {}

    def natural(element):
        path = element.asset or (element.frames[0] if element.frames else None)
        if not path:
            return None
        if path not in cache:
            size = QImageReader(str(path)).size()
            if not size.isValid():
                cache[path] = None
            else:
                spec = element.size_spec or AssetSizeSpec(
                    cols=element.sheet_cols, rows=element.sheet_rows)
                cache[path] = resolve(size.width(), size.height(),
                                      spec).natural
        return cache[path]

    return natural


def sweep_chart(sm_path: str) -> dict:
    """Compile one chart and measure it. Never raises - a chart that fails to
    compile is a RESULT (the sweep is a survey, not a gate), so one broken
    chart cannot end a multi-hour run."""
    from analysis.games.notitg import drawable_doc as dd
    from analysis.games.notitg.sim.producers import (
        compile_via_sim, wait_for_upgrade)

    started = time.monotonic()
    out = {'chart': sm_path, 'ok': False, 'seconds': 0.0}
    try:
        compiled = compile_via_sim(sm_path)
        if compiled is None:
            out['error'] = 'compile returned None'
            return out
        wait_for_upgrade(compiled)
        evaluator, id_maps, report = dd.build_static_doc(compiled)
        order = id_maps['element_order']
        rep = dd.element_parity_report(evaluator, order, _natural_lookup(),
                                       sample_times(compiled))
        out.update(
            ok=True,
            items=len(order),
            roles=dict(Counter(role for _el, role, _a in order)),
            skips=dict(report['element_skips']),
            images=report['images'],
            max_corner_err=round(float(rep['max_corner_err']), 4),
            all_ok=bool(rep['all_ok']),
            compared=sum(r['n_compared'] for r in rep['times']),
            unsized=sum(r['n_unsized'] for r in rep['times']),
        )
    except Exception as exc:  # noqa: BLE001 - a survey records failures
        out['error'] = f'{type(exc).__name__}: {exc}'
        out['traceback'] = traceback.format_exc(limit=6)
    finally:
        # In `finally` so an early return still carries a timing: a survey
        # that crashes formatting its own failure row is worse than useless.
        out['seconds'] = round(time.monotonic() - started, 1)
    return out


def find_charts(root: str) -> list:
    """Chart files under `root` that carry a storyboard worth measuring - a
    chart with no actor tree has no elements to compare."""
    charts = []
    for sm in sorted(Path(root).rglob('*.sm')):
        folder = sm.parent
        has_tree = any(folder.glob('*.xml')) or any(folder.glob('lua/*.xml')) \
            or (folder / 'default.lua').exists()
        if has_tree:
            charts.append(str(sm))
    return charts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/Yucky/Rhythm Games/Players/'
                                          'NotITG/Songs')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()

    charts = find_charts(args.root)
    print(f'{len(charts)} charts with storyboards under {args.root}')
    charts = charts[:args.limit] if args.limit > 0 else charts

    results = []
    skips = Counter()
    worst = 0.0
    for index, chart in enumerate(charts, 1):
        name = Path(chart).parent.name
        print(f'[{index}/{len(charts)}] {name} ...', flush=True)
        result = sweep_chart(chart)
        results.append(result)
        if result['ok']:
            skips.update(result['skips'])
            worst = max(worst, result['max_corner_err'])
            print(f'    {result["seconds"]}s  items={result["items"]} '
                  f'compared={result["compared"]} '
                  f'err={result["max_corner_err"]}px '
                  f'skips={result["skips"]}', flush=True)
        else:
            print(f'    {result["seconds"]}s  FAILED {result["error"]}',
                  flush=True)

    ok = [r for r in results if r['ok']]
    print(f'\n=== {len(ok)}/{len(results)} compiled, '
          f'worst corner error {worst}px ===')
    print('skipped element kinds across the sweep:')
    for kind, count in skips.most_common():
        print(f'  {kind:>16}  {count}')
    over = [r for r in ok if r['max_corner_err'] > 0.01]
    if over:
        print('\ncharts over 0.01px:')
        for r in sorted(over, key=lambda r: -r['max_corner_err']):
            print(f'  {r["max_corner_err"]:>10}px  {Path(r["chart"]).parent.name}')

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
