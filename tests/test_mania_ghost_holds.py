import numpy as np

from analysis.player.notes_model import NotesModel, link_miss_ghost_holds


def _model_for_holds(note_rows, note_cols, ghost_holds):
    model = NotesModel()
    model.noterows_list = list(note_rows)
    model.columns_list = list(note_cols)
    model.ln_tail_times = np.full(len(note_rows), 2.0, dtype=np.float64)
    model.ghost_hold_ln_heads_ms = np.array(
        [h[0] for h in ghost_holds], dtype=np.int64)
    model.ghost_hold_cols = np.array([h[1] for h in ghost_holds],
                                     dtype=np.int32)
    model.ghost_hold_press = np.array([h[2] for h in ghost_holds],
                                      dtype=np.float64)
    model.ghost_hold_release = np.array([h[3] for h in ghost_holds],
                                        dtype=np.float64)
    return model


def test_same_ln_only_first_miss_extends_from_head():
    model = _model_for_holds(
        note_rows=[1000, 1000],
        note_cols=[0, 0],
        ghost_holds=[
            (1000, 0, 1.100, 1.180),
            (1000, 0, 1.300, 1.380),
        ],
    )

    link_miss_ghost_holds(
        model,
        offsets=np.array([0.100, 0.300], dtype=np.float64),
        misses=np.array([True, True]),
        miss_pressed=np.array([True, True]),
    )

    assert model.miss_first_ghost_hold.tolist() == [0, -1]
    assert model.ghost_hold_extends_miss.tolist() == [True, False]
    assert model.miss_head_suppressed.tolist() == [False, True]


def test_different_lns_each_get_one_head_extension():
    model = _model_for_holds(
        note_rows=[1000, 1500],
        note_cols=[0, 0],
        ghost_holds=[
            (1000, 0, 1.100, 1.180),
            (1500, 0, 1.600, 1.680),
        ],
    )

    link_miss_ghost_holds(
        model,
        offsets=np.array([0.100, 0.100], dtype=np.float64),
        misses=np.array([True, True]),
        miss_pressed=np.array([True, True]),
    )

    assert model.miss_first_ghost_hold.tolist() == [0, 1]
    assert model.ghost_hold_extends_miss.tolist() == [True, True]
    assert model.miss_head_suppressed.tolist() == [False, False]
