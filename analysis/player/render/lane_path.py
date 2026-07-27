"""The curve an arrow travels down a lane, as one sampleable object.

Everything drawn in a lane sits somewhere on this curve: the receptor at
y_offset 0, a note head at its own offset, a hold BODY as the span between
its head and tail offsets, a hold TAIL at the end of that span, and a
travelpath as the whole visible range. Each of those used to be derived
separately - `ctx.receptor_offsets`, a `_NoteView`'s `lx`/`y`,
`ctx.hold_body_samples`, a tangent recomputed from that polyline - which is
four spellings of one question.

THE LAYERING THIS EXISTS FOR: this answers "what is this column like?" and
nothing else. A renderer depends on this interface and on no other renderer -
the receptor drawer does not need to know a hold body exists, and the ribbon
does not need to know where the cap ended up, because both ask the same
question and get consistent answers by construction.

That inverts what the pipeline used to do. The geometry producer PUSHED:
`note_mods` computed each consumer's answer in the consumer's own shape and
stashed it on the ctx under a name only that consumer read
(`receptor_offsets` a dict of arrays, `hold_body_samples` a candidate
position -> polyline), so the producer had to know the full list of who was
asking and every new drawn thing meant a new stash. Here a consumer PULLS,
and the producer supplies one hook it can answer for any (column, offset).

GAME-AGNOSTIC ON PURPOSE. Most games' lanes are a straight line, which is
this curve with no displacement, so the general form lives here and a game
with note mods supplies the bend. `straight` is that degenerate case and
needs nothing from a game but its lane geometry.

WHAT A SAMPLE CARRIES is where a thing is (x, y, z), how it is turned
(rotation, zoom), and how visible the LANE makes it there (alpha - the
hidden/sudden style gradients that are a function of position). Per-thing
appearance is not on the curve: a note's own glow, a receptor's own
visibility. NotITG's receptors are the sharp case - their visibility comes
from the dark family alone and the stealth terms the y-offset-0 pipeline
computes never apply to them (ReceptorArrowRow), even though their POSITION
comes through the same GetXPos as an arrow's. So the receptor consumer takes
x/y/rotation/zoom from here and ignores `alpha`, which is a consumer's call
to make, not a reason to keep the term off the curve that a hold body's
per-strip fade genuinely needs.

`y_offset` is the caller's own scroll space - whatever unit its notes are
already positioned in - and only the game's hooks give it meaning.
`straight` treats it as pixels below the receptor line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaneSample:
    """One point of the curve, in screen px / degrees / unitless."""

    x: float
    y: float
    z: float
    rotation_deg: float
    zoom: float
    alpha: float


@dataclass(frozen=True)
class LaneSamples:
    """Where a batch of (column, y_offset) pairs sits.

    Arrays, not one sample per call: a field asks for every receptor, head,
    tail and body subdivision it needs in ONE go, because the displacement
    behind them is a vectorized pipeline whose per-call cost dwarfs the
    per-row cost."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    rotation_deg: np.ndarray
    zoom: np.ndarray
    alpha: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def at(self, index: int) -> LaneSample:
        return LaneSample(float(self.x[index]), float(self.y[index]),
                          float(self.z[index]),
                          float(self.rotation_deg[index]),
                          float(self.zoom[index]), float(self.alpha[index]))

    def tangent_deg(self, first: int, last: int) -> float:
        """The heading in degrees from sample `first` to sample `last`, for
        a cap that has to follow the curve rather than the lane.

        A folded hold's end segment runs back UP the lane, and taking the
        heading rather than the lane's own direction is what makes the cap
        flip without a special case."""
        dx = float(self.x[last]) - float(self.x[first])
        dy = float(self.y[last]) - float(self.y[first])
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)))

    def rows(self, start: int, stop: int) -> 'LaneSamples':
        return LaneSamples(self.x[start:stop], self.y[start:stop],
                           self.z[start:stop],
                           self.rotation_deg[start:stop],
                           self.zoom[start:stop], self.alpha[start:stop])

    def box_filtered(self, taps: int) -> 'LaneSamples':
        """Every `taps` consecutive rows averaged into one.

        A span sampled coarsely aliases when the displacement's spatial
        period sits near the sample spacing: an unfiltered point sample
        turns a fine ripple into invented low-frequency lobes, while the
        engine, drawing strips every few px, converges to the local mean
        band. Averaging sub-samples spread across each cell is that band."""
        if taps <= 1:
            return self
        count = len(self.x) // taps

        def fold(values):
            return values.reshape(count, taps).mean(axis=1)

        return LaneSamples(fold(self.x), fold(self.y), fold(self.z),
                           fold(self.rotation_deg), fold(self.zoom),
                           fold(self.alpha))


class LanePath:
    """The lane curve for one frame, sampled in batches.

    Construct one per frame and sample it as many times as needed - it
    holds no per-sample state. The game supplies up to four hooks, all
    vectorized over equal-length arrays:

    - `lane_center(col) -> x`, the column's undisplaced screen center.
    - `note_y(cols, y_offsets) -> y`, where the undisplaced scroll axis
      puts that offset. Column-aware because a lane's axis can be placed
      per column (NotITG's reverse family slides one column's receptor to
      the mirrored line while its neighbour stays put).
    - `scroll_remap(y_offsets) -> y_offsets`, an optional reshape of the
      scroll axis itself (NotITG's accel family) applied BEFORE anything
      else, so `note_y` and `displace` cannot disagree about which offset
      they are answering for.
    - `displace(cols, y_offsets, note_beats, cell) -> (dx, dy, rotation_deg,
      zoom, z, alpha)`, the bend on top. None means a straight lane.

    `note_beats` is what is travelling, not just where: a displacement may
    depend on the thing's own beat (NotITG's dizzy spins a note by its
    beats-until-step, so a receptor - which has no step of its own - stays
    still). None lets the game pick its own default.

    `cell` is the y_offset spacing between neighbouring samples, 0 for a
    point query. A game whose displacement has fine spatial detail uses it
    to band-limit: what cannot be resolved at that spacing must not be
    point-sampled into invented shapes."""

    __slots__ = ('_lane_center', '_note_y', '_displace', '_scroll_remap')

    def __init__(self, lane_center, note_y, displace=None, scroll_remap=None):
        self._lane_center = lane_center
        self._note_y = note_y
        self._displace = displace
        self._scroll_remap = scroll_remap

    @property
    def displaces(self) -> bool:
        """Whether this lane bends at all. A consumer that has a cheaper
        drawing for a straight lane (a hold body as one rect rather than a
        ribbon) tests this rather than sampling and comparing."""
        return self._displace is not None

    def sample(self, cols, y_offsets, note_beats=None,
               cell: float = 0.0) -> LaneSamples:
        """Where each (column, y_offset) pair lands."""
        cols = np.asarray(cols, dtype=np.int64)
        offsets = np.asarray(y_offsets, dtype=np.float64)
        if self._scroll_remap is not None:
            offsets = np.asarray(self._scroll_remap(offsets),
                                 dtype=np.float64)
        x = self._centers(cols)
        y = np.asarray(self._note_y(cols, offsets), dtype=np.float64)
        zeros = np.zeros(len(cols), dtype=np.float64)
        ones = np.ones(len(cols), dtype=np.float64)
        if self._displace is None:
            return LaneSamples(x, y, zeros, zeros.copy(), ones, ones.copy())
        dx, dy, rot, zoom, z, alpha = self._displace(
            cols, offsets, note_beats, cell)
        return LaneSamples(x + np.asarray(dx, dtype=np.float64),
                           y + np.asarray(dy, dtype=np.float64),
                           np.asarray(z, dtype=np.float64),
                           np.asarray(rot, dtype=np.float64),
                           np.asarray(zoom, dtype=np.float64),
                           np.asarray(alpha, dtype=np.float64))

    def at(self, col: int, y_offset: float, note_beat=None) -> LaneSample:
        """One point, for a caller that genuinely wants one (a receptor).
        Prefer `sample` wherever a batch exists."""
        beats = None if note_beat is None else (float(note_beat),)
        return self.sample((col,), (y_offset,), beats).at(0)

    def between(self, col: int, start: float, end: float, count: int,
                note_beat=None, cell: float = 0.0,
                taps: int = 1) -> LaneSamples:
        """The span from `start` to `end` as `count` points, endpoints
        included - a hold body, or a travelpath over the visible range.

        The endpoints land exactly on `start` and `end`, so a body stays
        attached to the head and tail that were placed independently."""
        beats = None if note_beat is None else (float(note_beat),)
        return self.spans((col,), (start,), (end,), (count,), beats,
                          cell, taps)[0]

    def spans(self, cols, starts, ends, counts, note_beats=None,
              cell: float = 0.0, taps: int = 1) -> list:
        """One `LaneSamples` per span, all sampled in ONE displacement
        call. This is the shape a field actually asks in: every visible
        hold's body is a span, and evaluating them one at a time pays the
        vectorized pipeline's fixed cost per hold instead of per frame."""
        cols = np.asarray(cols, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        col_parts, offset_parts, beat_parts, lengths = [], [], [], []
        for i in range(len(cols)):
            count = max(2, int(counts[i]))
            fractions = _sub_sampled(count, taps)
            offset_parts.append(
                starts[i] + fractions * (ends[i] - starts[i]))
            col_parts.append(np.full(len(fractions), cols[i], dtype=np.int64))
            if note_beats is not None:
                beat_parts.append(np.full(len(fractions), note_beats[i]))
            lengths.append(count)
        if not lengths:
            return []
        batch = self.sample(
            np.concatenate(col_parts), np.concatenate(offset_parts),
            np.concatenate(beat_parts) if beat_parts else None, cell)
        out, cursor = [], 0
        for count in lengths:
            width = count * max(1, taps)
            out.append(batch.rows(cursor, cursor + width).box_filtered(taps))
            cursor += width
        return out

    def _centers(self, cols) -> np.ndarray:
        """`lane_center` per sample, called once per distinct column: a
        span carries hundreds of samples of the same lane."""
        columns = np.unique(cols)
        table = np.zeros(int(columns.max()) + 1, dtype=np.float64)
        for col in columns:
            table[col] = float(self._lane_center(int(col)))
        return table[cols]


def _sub_sampled(count: int, taps: int) -> np.ndarray:
    """`count` evenly spaced fractions of a span, each replaced by `taps`
    sub-positions spread across its own cell.

    The sub-positions of an END cell deliberately run a hair past 0 and 1:
    the displacement is a pure function of offset, defined everywhere, and
    keeping each cell's samples centred on its point is what keeps the
    filtered endpoint attached to the head or tail placed there."""
    fractions = np.linspace(0.0, 1.0, count)
    if taps <= 1:
        return fractions
    cell = 1.0 / max(count - 1, 1)
    jitter = (np.arange(taps) / taps - 0.5 + 0.5 / taps) * cell
    return (fractions[:, None] + jitter[None, :]).ravel()


def straight(lane_center, note_y) -> LanePath:
    """A lane with no displacement - the case every game without note mods
    has, and the one NotITG's bent path degenerates to when its mods are
    all at zero. `note_y` takes the offsets alone; the column cannot move
    an axis that does not bend."""
    return LanePath(lane_center, lambda cols, offsets: note_y(offsets))
