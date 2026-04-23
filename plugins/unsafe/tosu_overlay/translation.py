"""Back-compat shim that merges v1 + v2 schemas into a single state dict.

This module exists so callers (``view.py`` and legacy tests) can import
``build_tosu_state`` / ``build_precise_state`` / ``parse_filter_message``
/ ``prune_to_filters`` from one place without worrying about which
schema their target overlay uses. The shim is a pure orchestrator; all
real translation logic lives in the per-schema modules:

    translation_v2.py     -- modern ``api_v2`` / ``api_v2_precise``
    translation_v1.py     -- legacy ``api_v1`` (menu/gameplay)
    translation_common.py -- filter parsing, pruning, pure helpers

Since the shim fans every push out to every live WebSocket regardless
of URL (see ``shim.js``), merging both schemas into one payload means
overlays built against either API read what they expect.
"""
from __future__ import annotations

from plugins.unsafe.tosu_overlay import translation_v1, translation_v2
from plugins.unsafe.tosu_overlay.translation_common import (
    _flatten_filter_list,
    _safe,
    acc_percent as _acc_percent,
    client_string as _client_string,
    hits_dict as _hits_dict,
    mode_for_game as _mode_for_game,
    parse_filter_message,
    prune_to_filters,
    unstable_rate_from_errors,
)

# Re-exports under their old names so existing tests keep importing
# them from ``translation`` rather than ``translation_common``.
__all__ = [
    'build_tosu_state',
    'build_precise_state',
    'parse_filter_message',
    'prune_to_filters',
    'unstable_rate_from_errors',
    # Legacy private helpers referenced by tests.
    '_flatten_filter_list',
    '_safe',
    '_hits_dict',
    '_acc_percent',
    '_mode_for_game',
    '_client_string',
]


def build_tosu_state(game_state) -> dict:
    """Merged v1 + v2 payload. v2 keys dominate the outer object; the
    v1 builder's ``menu`` / ``gameplay`` entries are added alongside so
    legacy overlays find their paths too.

    The v1 ``menu.mods.num`` is filled in from v2's ``play.mods.number``
    (the only place we actually derive the bitfield), so overlays that
    read ``data.menu.mods.num`` get the right value too.
    """
    state = translation_v2.build_state(game_state)
    v1 = translation_v1.build_state(game_state)

    # Patch the legacy-mods bitfield from v2's computed value.
    v1['menu']['mods']['num'] = state['play']['mods']['number']

    state['menu'] = v1['menu']
    state['gameplay'] = v1['gameplay']
    return state


def build_precise_state(game_state) -> dict:
    return translation_v2.build_precise(game_state)
