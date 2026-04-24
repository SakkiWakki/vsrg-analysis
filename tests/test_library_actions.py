"""Tests for the library-toolbar action registry.

The registry is the backend for plugin-contributed buttons on the
library tab. Tests exercise add/replace-by-key semantics, per-module
cleanup, listener notification, and listener error isolation — each
property is load-bearing for a plugin to safely add or remove buttons
without the UI going stale or a badly-written listener taking the
registry down."""
from __future__ import annotations

from analysis.gui.library.actions import (LibraryActionRegistry,
                                          reset_for_tests, get_registry)


def test_add_returns_action_and_stores_it():
    r = LibraryActionRegistry()
    a = r.add('Foo', lambda: None, module='m')
    assert a.label == 'Foo'
    assert r.actions() == [a]


def test_add_replaces_existing_same_key():
    r = LibraryActionRegistry()
    called = []
    r.add('Foo', lambda: called.append('old'), key='k')
    r.add('Foo', lambda: called.append('new'), key='k')
    assert len(r.actions()) == 1
    r.actions()[0].callback()
    assert called == ['new']


def test_clear_module_removes_only_that_modules_actions():
    r = LibraryActionRegistry()
    r.add('A', lambda: None, module='x')
    r.add('B', lambda: None, module='y')
    r.add('C', lambda: None, module='x')
    removed = r.clear_module('x')
    assert removed == 2
    labels = [a.label for a in r.actions()]
    assert labels == ['B']


def test_listeners_fire_on_add_and_clear():
    r = LibraryActionRegistry()
    hits = []
    r.subscribe(lambda: hits.append(1))
    r.add('A', lambda: None, module='m')
    assert hits == [1]
    r.clear_module('m')
    assert hits == [1, 1]


def test_clear_module_with_no_matches_does_not_notify():
    r = LibraryActionRegistry()
    hits = []
    r.subscribe(lambda: hits.append(1))
    r.clear_module('nope')
    assert hits == []


def test_unsubscribe_stops_notifications():
    r = LibraryActionRegistry()
    hits = []
    unsub = r.subscribe(lambda: hits.append(1))
    r.add('A', lambda: None)
    unsub()
    r.add('B', lambda: None)
    assert hits == [1]


def test_failing_listener_does_not_break_registry():
    """One bad subscriber shouldn't prevent subsequent listeners, or
    the add itself, from completing."""
    r = LibraryActionRegistry()
    good_hits = []
    r.subscribe(lambda: (_ for _ in ()).throw(RuntimeError('boom')))
    r.subscribe(lambda: good_hits.append(1))
    r.add('A', lambda: None)
    assert good_hits == [1]
    assert len(r.actions()) == 1


def test_empty_label_rejected():
    r = LibraryActionRegistry()
    try:
        r.add('   ', lambda: None)
    except ValueError:
        return
    assert False, 'expected ValueError on empty label'


def test_non_callable_rejected():
    r = LibraryActionRegistry()
    try:
        r.add('A', 42)
    except TypeError:
        return
    assert False, 'expected TypeError on non-callable'


def test_get_registry_returns_singleton():
    reset_for_tests()
    a = get_registry()
    b = get_registry()
    assert a is b
    reset_for_tests()
