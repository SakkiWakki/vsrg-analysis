"""Shared translation helpers for both tosu v1 and v2 schemas.

The per-schema builders (``translation_v1``, ``translation_v2``) each
construct a dict whose keys match the respective tosu websocket
protocol. Both builders share:

  - ``parse_filter_message`` + ``_flatten_filter_list`` for inbound
    ``applyFilters:`` payloads.
  - ``prune_to_filters`` for trimming the outbound state dict.
  - ``_safe`` for tolerant GameState access.
  - Small pure helpers (``_hits_dict``, ``_acc_percent``, etc.).

None of these helpers touch Player directly; all go through the
unified Component API (``GameState`` + chart snapshot dataclasses).
"""
from __future__ import annotations

import json
import math

from analysis.components.api import DataNotAvailable


# ---------------------------------------------------------------------------
# Inbound: parse applyFilters payload
# ---------------------------------------------------------------------------

def parse_filter_message(raw: str) -> frozenset[str] | None:
    """Return a frozenset of dotted field paths from a ``ws.send()``
    payload, or ``None`` if the message is not an applyFilters command.
    The same parser handles both v1 and v2 overlays; schema is
    distinguished by the field names the overlay requests."""
    if not raw.startswith('applyFilters:'):
        return None
    body = raw[len('applyFilters:'):]
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    return _flatten_filter_list(data)


def _flatten_filter_list(data, prefix: str = '') -> frozenset[str]:
    """Normalize both filter-list shapes into dotted paths.

    Handles recursion so nested objects like
        [{field: "play", keys: [{field: "mods", keys: ["name"]}, "score"]}]
    expand to {"play.mods.name", "play.score"}.
    """
    if not isinstance(data, list):
        return frozenset()
    out: set[str] = set()
    for item in data:
        match item:
            case str():
                out.add(f'{prefix}.{item}' if prefix else item)
            case {'field': str(field), 'keys': list(keys)}:
                child_prefix = f'{prefix}.{field}' if prefix else field
                for k in keys:
                    if isinstance(k, str):
                        out.add(f'{child_prefix}.{k}')
                    elif isinstance(k, dict):
                        out |= _flatten_filter_list([k], child_prefix)
            case _:
                pass
    return frozenset(out)


# ---------------------------------------------------------------------------
# Filter pruning
# ---------------------------------------------------------------------------

def prune_to_filters(state: dict, filters: frozenset[str]) -> dict:
    """Prune ``state`` to only include paths requested by the overlay.

    Returns the state unchanged if filters is empty. A key is kept iff
    its dotted path is equal to, a prefix of, or has-as-prefix any path
    in the filter set. Works the same way for v1 and v2 dicts.
    """
    if not filters:
        return state
    return _prune_node(state, filters, prefix='')


def _prune_node(node, filters: frozenset[str], prefix: str):
    if not isinstance(node, dict):
        return node
    out = {}
    for key, val in node.items():
        path = f'{prefix}.{key}' if prefix else key
        if _path_wanted(path, filters):
            out[key] = _prune_node(val, filters, path)
    return out


def _path_wanted(path: str, filters: frozenset[str]) -> bool:
    for f in filters:
        if path == f or path.startswith(f + '.') or f.startswith(path + '.'):
            return True
    return False


# ---------------------------------------------------------------------------
# Safe GameState access
# ---------------------------------------------------------------------------

def _safe(fn, default):
    """Call a zero-arg GameState method; return ``default`` on
    DataNotAvailable or AttributeError (backend missing the method)."""
    try:
        return fn()
    except (DataNotAvailable, AttributeError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Shared numeric / string helpers
# ---------------------------------------------------------------------------

def hits_dict(counts: dict[str, int]) -> dict:
    """Canonical hit-counts dict used by both v1 (gameplay.hits) and v2
    (play.hits). Accepts our internal judgment names or tosu's
    perfect/great/good/bad aliases."""
    return {
        '300':  counts.get('300',  counts.get('perfect', 0)),
        '200':  counts.get('200',  counts.get('great', 0)),
        '100':  counts.get('100',  counts.get('good', 0)),
        '50':   counts.get('50',   counts.get('bad', 0)),
        '0':    counts.get('miss', 0),
        'geki': counts.get('geki', counts.get('300g', 0)),
        'katu': counts.get('katu', counts.get('200', 0)),
        'sliderBreaks': 0,
    }


def acc_percent(counts: dict[str, int]) -> float:
    """osu!mania weighted accuracy -> percent (0..100)."""
    if not counts:
        return 0.0
    weights = {
        'geki': 325, '300g': 325, '300': 300, 'perfect': 325,
        'katu': 200, '200': 200, 'great': 300, '100': 100,
        'good': 100, '50': 50, 'bad': 50, 'miss': 0,
    }
    weighted = sum(weights.get(k, 0) * v for k, v in counts.items())
    total_hits = sum(v for k, v in counts.items() if k != 'sliderBreaks')
    if total_hits == 0:
        return 0.0
    return round(weighted / (total_hits * 325.0) * 100.0, 4)


def mode_for_game(game: str) -> tuple[int, str]:
    g = (game or '').lower()
    if g in ('osu', 'osumania', 'mania'):
        return 3, 'mania'
    if g == 'etterna':
        return 3, 'mania'        # etterna overlay treated as mania
    if g == 'taiko':
        return 1, 'taiko'
    if g in ('catch', 'fruits'):
        return 2, 'fruits'
    return 3, 'mania'


def client_string() -> str:
    return 'tosu-shim/vsrg-analysis'


def unstable_rate_from_errors(hit_errors_ms) -> float:
    """Fallback UR calc when the backend doesn't expose unstable_rate."""
    n = len(hit_errors_ms)
    if n < 2:
        return 0.0
    mean = sum(hit_errors_ms) / n
    var = sum((x - mean) ** 2 for x in hit_errors_ms) / n
    return round(10.0 * math.sqrt(var), 4)


# ---------------------------------------------------------------------------
# State enum mappings (differ between v1 and v2)
# ---------------------------------------------------------------------------

def v2_state_name(paused: bool) -> str:
    return 'Menu' if paused else 'Playing'


def v2_state_number(paused: bool) -> int:
    # tosu v2 state enum: 0 Menu, 2 Playing, 5 SongSelect, 7 ResultsScreen.
    return 0 if paused else 2


def v1_state_number(paused: bool) -> int:
    # osu! stable OsuStatus enum (what v1 overlays read as data.menu.state):
    #   0 Menu, 2 Playing, 5 SongSelect, 7 ResultsScreen, ...
    return 0 if paused else 2
