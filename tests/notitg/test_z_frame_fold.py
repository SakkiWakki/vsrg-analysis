"""SetDrawByZPosition cross-stream sort: elements inside a flagged
frame's tree-index span fold into the SAME SortSpan run as the frame's
field instances (FFF weaves its player proxies through six stripe
sprites - instance-only runs left the stripes in fixed tree order)."""
from analysis.games.notitg.drawable_doc import (_FIELD_BAND, _Unit,
                                                _fold_z_runs, _span_group)


def _unit(tree_index, kind, group=None, payload=None):
    return _Unit(*_FIELD_BAND, tree_index, tree_index, kind,
                 payload or {}, group=group)


def test_span_group_maps_tree_indexes():
    spans = [(29, 28, 36)]
    assert _span_group(spans, 30) == 29
    assert _span_group(spans, 28) == 29
    assert _span_group(spans, 37) is None
    assert _span_group(spans, None) is None


def test_consecutive_grouped_units_fold_across_streams():
    units = [
        _unit(10, 'element'),
        _unit(28, 'instance', group=29),
        _unit(29, 'element', group=29),
        _unit(30, 'element', group=29),
        _unit(40, 'instance'),
    ]
    folded = _fold_z_runs(units)
    kinds = [u.kind for u in folded]
    assert kinds == ['element', 'z_run', 'instance']
    run = folded[1]
    assert [m.kind for m in run.payload] == ['instance', 'element',
                                             'element']
    assert run.group == 29


def test_separate_groups_stay_separate_runs():
    units = [_unit(1, 'instance', group=5), _unit(2, 'instance', group=7)]
    folded = _fold_z_runs(units)
    assert [u.kind for u in folded] == ['z_run', 'z_run']
    assert [u.group for u in folded] == [5, 7]
