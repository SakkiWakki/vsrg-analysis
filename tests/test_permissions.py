"""Safety tests for the plugin URL permission store.

Covers: decision persistence, round-trip, per-plugin isolation,
URL key collision resistance, and the http_get permission gate.
"""
from __future__ import annotations

import pytest

from analysis.config import get_config
from analysis.plugins.permissions import Decision, clear, record, stored
from analysis.plugins.host_api import (
    NetworkAccessDenied,
    http_get,
    set_invoke_on_main,
    set_permission_dialog,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _clean(*pairs):
    """Remove test-written permission keys from config."""
    for plugin_key, url in pairs:
        clear(plugin_key, url)


# ── Decision store ────────────────────────────────────────────────────

def test_no_stored_decision_returns_none():
    _clean(('test:plugin', 'http://example.com'))
    assert stored('test:plugin', 'http://example.com') is None


def test_record_always_persists_and_is_readable():
    _clean(('test:plugin', 'http://example.com'))
    record('test:plugin', 'http://example.com', Decision.ALWAYS)
    assert stored('test:plugin', 'http://example.com') == Decision.ALWAYS
    _clean(('test:plugin', 'http://example.com'))


def test_record_never_persists_and_is_readable():
    _clean(('test:plugin', 'http://example.com'))
    record('test:plugin', 'http://example.com', Decision.NEVER)
    assert stored('test:plugin', 'http://example.com') == Decision.NEVER
    _clean(('test:plugin', 'http://example.com'))


def test_clear_removes_stored_decision():
    record('test:plugin', 'http://example.com', Decision.ALWAYS)
    clear('test:plugin', 'http://example.com')
    assert stored('test:plugin', 'http://example.com') is None


def test_decisions_are_isolated_per_plugin():
    _clean(('plugin:a', 'http://x.com'), ('plugin:b', 'http://x.com'))
    record('plugin:a', 'http://x.com', Decision.ALWAYS)
    assert stored('plugin:a', 'http://x.com') == Decision.ALWAYS
    assert stored('plugin:b', 'http://x.com') is None
    _clean(('plugin:a', 'http://x.com'), ('plugin:b', 'http://x.com'))


def test_decisions_are_isolated_per_url():
    _clean(('test:plugin', 'http://a.com'), ('test:plugin', 'http://b.com'))
    record('test:plugin', 'http://a.com', Decision.ALWAYS)
    assert stored('test:plugin', 'http://a.com') == Decision.ALWAYS
    assert stored('test:plugin', 'http://b.com') is None
    _clean(('test:plugin', 'http://a.com'), ('test:plugin', 'http://b.com'))


def test_url_with_dots_and_slashes_does_not_collide():
    url_a = 'http://evil.com/foo'
    url_b = 'http://evil_com_foo'
    _clean(('test:plugin', url_a), ('test:plugin', url_b))
    record('test:plugin', url_a, Decision.ALWAYS)
    record('test:plugin', url_b, Decision.NEVER)
    assert stored('test:plugin', url_a) == Decision.ALWAYS
    assert stored('test:plugin', url_b) == Decision.NEVER
    _clean(('test:plugin', url_a), ('test:plugin', url_b))


def test_decision_enum_values_are_stable():
    assert Decision.ALWAYS.value == 'always'
    assert Decision.NEVER.value == 'never'


# ── http_get permission gate ──────────────────────────────────────────

def test_http_get_raises_when_never_stored():
    url = 'http://never.example.com'
    _clean(('test:plugin', url))
    record('test:plugin', url, Decision.NEVER)
    try:
        with pytest.raises(NetworkAccessDenied):
            http_get('test:plugin', url)
    finally:
        _clean(('test:plugin', url))


def test_http_get_denies_when_dialog_returns_deny_once():
    url = 'http://deny-once.example.com'
    _clean(('test:plugin', url))

    set_invoke_on_main(lambda fn: fn())
    set_permission_dialog(lambda plugin_key, u: 'deny_once')
    try:
        with pytest.raises(NetworkAccessDenied):
            http_get('test:plugin', url)
        # deny_once must NOT persist
        assert stored('test:plugin', url) is None
    finally:
        set_permission_dialog(None)
        _clean(('test:plugin', url))


def test_http_get_stores_always_when_dialog_returns_always():
    url = 'http://always.example.com'
    _clean(('test:plugin', url))

    set_invoke_on_main(lambda fn: fn())
    # Simulate "always" then immediately raise so we don't do real HTTP
    call_count = {'n': 0}
    def _dialog(plugin_key, u):
        call_count['n'] += 1
        return 'always'
    set_permission_dialog(_dialog)
    try:
        with pytest.raises(Exception):
            http_get('test:plugin', url)
        assert stored('test:plugin', url) == Decision.ALWAYS
        assert call_count['n'] == 1
    finally:
        set_permission_dialog(None)
        _clean(('test:plugin', url))


def test_http_get_stores_never_when_dialog_returns_never():
    url = 'http://store-never.example.com'
    _clean(('test:plugin', url))

    set_invoke_on_main(lambda fn: fn())
    set_permission_dialog(lambda plugin_key, u: 'never')
    try:
        with pytest.raises(NetworkAccessDenied):
            http_get('test:plugin', url)
        assert stored('test:plugin', url) == Decision.NEVER
    finally:
        set_permission_dialog(None)
        _clean(('test:plugin', url))


def test_http_get_skips_dialog_when_always_stored():
    url = 'http://skip-dialog.example.com'
    _clean(('test:plugin', url))
    record('test:plugin', url, Decision.ALWAYS)

    dialog_called = {'v': False}
    set_invoke_on_main(lambda fn: fn())
    set_permission_dialog(lambda pk, u: (_ for _ in ()).throw(
        AssertionError('dialog should not be shown')))
    try:
        # Will fail with a network error (not a permission error) since
        # the URL doesn't exist -- that's fine, it proves the dialog was
        # skipped.
        with pytest.raises(Exception) as exc_info:
            http_get('test:plugin', url)
        assert not isinstance(exc_info.value, NetworkAccessDenied)
    finally:
        set_permission_dialog(None)
        _clean(('test:plugin', url))


def test_http_get_denies_without_dialog_when_no_qt_host():
    url = 'http://no-host.example.com'
    _clean(('test:plugin', url))
    set_permission_dialog(None)
    set_invoke_on_main(lambda fn: fn())
    try:
        with pytest.raises(NetworkAccessDenied):
            http_get('test:plugin', url)
    finally:
        _clean(('test:plugin', url))
