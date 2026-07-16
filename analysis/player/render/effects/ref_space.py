"""fluXis's gameplay reference draw space.

Event coordinates (playfield moves, shakes, camera moves) are authored
in pixels at this DrawSizePreservingFillContainer target resolution;
effects scale them by `chart_rect / ref` so authored motion stays
screen-proportional at any window size.
"""

REF_W = 1366.0
REF_H = 768.0
