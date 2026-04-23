"""Per-plugin URL permission store.

Tracks user decisions about which URLs a plugin is allowed to access.
Decisions are persisted per-plugin under
``plugins.<key>.permissions.<escaped_url>`` in the shared config store.

The decision dialog is not implemented here -- this module only stores
and retrieves decisions. The host's Qt layer is responsible for showing
the dialog and passing the result back via :func:`record`.

Examples:
    'always'     -- user chose "Always allow"; future calls skip the dialog
    'never'      -- user chose "Never allow"; future calls raise immediately
    None         -- no stored decision; the dialog should be shown
"""
from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    ALWAYS = 'always'
    NEVER  = 'never'


def _perm_path(plugin_key: str, url: str) -> str:
    from analysis.plugins.host_api import _escape_key
    # Encode characters that are either config-path separators (.) or
    # URL structure (/:?) using %XX so two different URLs can never
    # produce the same key.
    safe_url = url.replace('%', '%25').replace('.', '%2E').replace('/', '%2F').replace(':', '%3A')
    return f'plugins.{_escape_key(plugin_key)}.permissions.{safe_url}'


def stored(plugin_key: str, url: str) -> Decision | None:
    """Return the persisted decision for this plugin+URL, or None if the
    user has never made a permanent choice (dialog must be shown)."""
    from analysis.config import get_config
    raw = get_config().get(_perm_path(plugin_key, url))
    try:
        return Decision(raw) if raw is not None else None
    except ValueError:
        return None


def record(plugin_key: str, url: str, decision: Decision) -> None:
    """Persist a user decision. Only ALWAYS and NEVER are stored
    (once/deny-once are ephemeral and handled by the caller)."""
    from analysis.config import get_config
    get_config().set(_perm_path(plugin_key, url), decision.value)


def clear(plugin_key: str, url: str) -> None:
    """Remove any stored decision for this plugin+URL."""
    from analysis.config import get_config
    get_config().delete(_perm_path(plugin_key, url))
