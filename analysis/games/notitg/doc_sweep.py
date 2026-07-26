"""Sweep NotITG charts through both drawable-doc parity harnesses.

Two charts is not a corpus. This compiles every chart it is pointed at,
builds the static doc, and reports, per chart:

  - ELEMENT PLACEMENT: the drawn-quad corner error against the legacy
    painter, in design pixels (`drawable_doc.element_parity_report`).
  - FIELD PLACEMENT: the same measure for field instances, against
    `NotitgFieldInstances` (`drawable_doc.field_parity_report`), plus which
    instances each side draws that the other does not.
  - COVERAGE: the element kinds the doc SKIPS, and any field-link prop the
    chart POKES that the doc's link table does not carry. A placement number
    only describes what the doc emitted; it cannot see what was never
    emitted at all, which is how 118 missing rects hid behind a 0.001px
    result on gat 1.

Field parity is measured with the CAPTURED-notefield representation
(`VSRG_DRAWABLE_NOTES=0`), because that is the representation the legacy
effect models: one blit per instance. Under the default inline-notes path a
field copy re-renders the notes as many items and has no single quad to
compare, so the sweep would silently measure almost nothing.

Usage:
    python -m analysis.games.notitg.doc_sweep [--limit N] [--root DIR] [--json OUT]

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


def _uncarried_link_props(compiled) -> dict:
    """`prop -> link count` for every field-link prop the chart POKES that the
    doc's link table does not carry.

    The gap a placement harness structurally cannot see: a prop the doc never
    reads is not a difference in where an instance lands, it is a missing
    input, and the quad comes out plausible. `_LINK_PROP_ORDER` omitting the
    out-of-plane terms discarded every 3D field transform and measured clean
    while doing it."""
    from analysis.games.notitg import drawable_doc as dd
    from analysis.games.notitg import field_compose as fc

    carried = {prop for _param, prop in dd._LINK_PROP_ORDER}
    carried |= {'rotation_order', 'fov'}
    provider = compiled.get('field_instances')
    instances = provider() if callable(provider) else (provider or [])
    poked: Counter = Counter()
    for inst in instances:
        for link in inst['transform']._links:
            for prop, rest in fc._LINK_RESTS.items():
                if prop not in carried and dd._moves_off(link.get(prop), rest):
                    poked[prop] += 1
    return dict(poked)


def _field_measure(compiled, times) -> dict:
    """Field-instance parity for one chart, under the captured-notefield
    representation the legacy effect models (see the module note)."""
    from analysis.games.notitg import drawable_doc as dd

    previous = os.environ.get('VSRG_DRAWABLE_NOTES')
    os.environ['VSRG_DRAWABLE_NOTES'] = '0'
    try:
        evaluator, id_maps, _report = dd.build_static_doc(compiled)
        order = id_maps['instance_order']
        rep = dd.field_parity_report(evaluator, compiled, order, times)
    finally:
        if previous is None:
            del os.environ['VSRG_DRAWABLE_NOTES']
        else:
            os.environ['VSRG_DRAWABLE_NOTES'] = previous
    return {
        'instances': len(order),
        'field_compared': rep['n_compared'],
        'field_3d': rep['n_projected'],
        'field_chained': rep['n_chain_folded'],
        'field_missing': rep['n_missing'],
        'field_extra': rep['n_extra'],
        'field_err': round(float(rep['max_corner_err']), 4),
        'field_ok': bool(rep['all_ok']),
    }


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
        times = sample_times(compiled)
        evaluator, id_maps, report = dd.build_static_doc(compiled)
        order = id_maps['element_order']
        rep = dd.element_parity_report(evaluator, order, _natural_lookup(),
                                       times)
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
            uncarried=_uncarried_link_props(compiled),
            **_field_measure(compiled, times),
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


def _print_over(results, key: str, label: str, bar: float = 0.01) -> None:
    """The charts whose `key` error clears `bar`, worst first."""
    over = [r for r in results if r[key] > bar]
    if not over:
        return
    print(f'\n{label} placement over {bar}px:')
    for r in sorted(over, key=lambda r: -r[key]):
        print(f'  {r[key]:>10}px  {Path(r["chart"]).parent.name}')


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
    uncarried = Counter()
    worst = worst_field = 0.0
    for index, chart in enumerate(charts, 1):
        name = Path(chart).parent.name
        print(f'[{index}/{len(charts)}] {name} ...', flush=True)
        result = sweep_chart(chart)
        results.append(result)
        if result['ok']:
            skips.update(result['skips'])
            uncarried.update(result['uncarried'])
            worst = max(worst, result['max_corner_err'])
            worst_field = max(worst_field, result['field_err'])
            print(f'    {result["seconds"]}s  items={result["items"]} '
                  f'compared={result["compared"]} '
                  f'err={result["max_corner_err"]}px '
                  f'skips={result["skips"]}', flush=True)
            print(f'      fields: {result["instances"]} instances '
                  f'compared={result["field_compared"]} '
                  f'3d={result["field_3d"]} '
                  f'chained={result["field_chained"]} '
                  f'missing={result["field_missing"]} '
                  f'extra={result["field_extra"]} '
                  f'err={result["field_err"]}px', flush=True)
        else:
            print(f'    {result["seconds"]}s  FAILED {result["error"]}',
                  flush=True)

    ok = [r for r in results if r['ok']]
    print(f'\n=== {len(ok)}/{len(results)} compiled, worst corner error: '
          f'elements {worst}px, fields {worst_field}px ===')
    print('skipped element kinds across the sweep:')
    for kind, count in skips.most_common():
        print(f'  {kind:>16}  {count}')
    print('field-link props the doc does NOT carry, by links poking them:')
    for prop, count in uncarried.most_common():
        print(f'  {prop:>16}  {count}')
    _print_over(ok, 'max_corner_err', 'element')
    _print_over(ok, 'field_err', 'field')
    dropped = [r for r in ok if r['field_missing'] or r['field_extra']]
    if dropped:
        print('\ncharts where the two sides draw different instances:')
        for r in dropped:
            print(f'  missing={r["field_missing"]:>4} '
                  f'extra={r["field_extra"]:>4}  '
                  f'{Path(r["chart"]).parent.name}')

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
