"""Scrollable Quaver-style note + press visualizer. Keycount inferred from replay."""


def build(replay, game='etterna', on_play=None, od=None, judge=None, **_):
    from analysis.gui.note_viz_tab import NoteVizTab
    kwargs = {}
    if od is not None:
        kwargs['od'] = od
    if judge is not None:
        kwargs['judge'] = judge
    w = NoteVizTab(replay, game=game, on_play=on_play, **kwargs)
    w._has_play_btn = on_play is not None
    return w


def register(add):
    add('Note visualizer (scrollable)', build, category='widget')
