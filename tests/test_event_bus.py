"""Tests for the synchronous event bus."""
from __future__ import annotations

import pytest

from analysis.player.input.events import EventBus


def test_emit_with_no_handlers_is_noop():
    bus = EventBus()
    bus.emit('whatever')
    bus.emit('whatever', payload='data')


def test_on_emit_no_payload():
    bus = EventBus()
    calls = []
    bus.on('ping', lambda: calls.append(None))
    bus.emit('ping')
    assert calls == [None]


def test_on_emit_scalar_payload():
    bus = EventBus()
    calls = []
    bus.on('pong', lambda p: calls.append(p))
    bus.emit('pong', 42)
    assert calls == [42]


def test_on_emit_tuple_payload_is_spread():
    """Tuple payloads unpack as positional args ; lets handlers declare
    their real signature instead of tuple-unpacking by hand."""
    bus = EventBus()
    calls = []
    bus.on('act', lambda kind, data: calls.append((kind, data)))
    bus.emit('act', ('toggle_sv', None))
    assert calls == [('toggle_sv', None)]


def test_multiple_handlers_called_in_registration_order():
    bus = EventBus()
    order = []
    bus.on('k', lambda: order.append('a'))
    bus.on('k', lambda: order.append('b'))
    bus.on('k', lambda: order.append('c'))
    bus.emit('k')
    assert order == ['a', 'b', 'c']


def test_off_unsubscribes():
    bus = EventBus()
    calls = []
    sub = bus.on('k', lambda: calls.append(1))
    assert bus.off(sub) is True
    bus.emit('k')
    assert calls == []


def test_off_is_idempotent():
    bus = EventBus()
    sub = bus.on('k', lambda: None)
    assert bus.off(sub) is True
    assert bus.off(sub) is False


def test_handler_exception_does_not_stop_chain():
    """A throwing handler must not prevent later handlers from firing ;
    otherwise one buggy plugin could silently break the whole app."""
    bus = EventBus()
    later_called = []

    def boom():
        raise RuntimeError('boom')

    bus.on('k', boom)
    bus.on('k', lambda: later_called.append(True))
    bus.emit('k')
    assert later_called == [True]


def test_handler_can_unsubscribe_during_dispatch():
    """Snapshot the handler list before dispatch so a handler that calls
    ``off`` doesn't mutate the iterable underneath us."""
    bus = EventBus()
    calls = []
    sub = None

    def first():
        calls.append('first')
        bus.off(sub)

    sub = bus.on('k', first)
    bus.on('k', lambda: calls.append('second'))
    bus.emit('k')
    # Both fire this tick; second emit only has 'second'.
    assert calls == ['first', 'second']
    bus.emit('k')
    assert calls == ['first', 'second', 'second']


def test_handler_can_subscribe_during_dispatch():
    """New subscribers added mid-dispatch must not fire this tick."""
    bus = EventBus()
    calls = []

    def outer():
        calls.append('outer')
        bus.on('k', lambda: calls.append('inner'))

    bus.on('k', outer)
    bus.emit('k')
    assert calls == ['outer']
    bus.emit('k')
    assert calls == ['outer', 'outer', 'inner']


def test_kinds_are_isolated():
    bus = EventBus()
    a = []
    b = []
    bus.on('a', lambda: a.append(1))
    bus.on('b', lambda: b.append(1))
    bus.emit('a')
    assert a == [1]
    assert b == []
