"""Minimal synchronous event bus.

Replaces ad-hoc ``_xxx_listeners`` lists on ``Player``. Handlers are
called synchronously on ``emit`` in registration order; exceptions in
one handler don't stop the others. Unsubscribe via the returned handle.

Kinds are plain strings. Canonical ones used by the host today:

  * ``scroll_changed`` — no payload. Fired when the scroll mode, scroll
    value, rate, or active game change via HUD controls. Subscribers
    typically re-sync audio and persist settings.
  * ``hud_action`` — payload ``(action, data)``. Fired for HUD toggles
    whose logical state lives outside the player (audio pitch-correct,
    QSettings-backed toggles, transient overlays).

New kinds should be documented here when added so plugin authors have a
single place to learn what they can subscribe to.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Subscription:
    """Opaque handle returned by ``EventBus.on``. Pass back to
    ``EventBus.off`` to unsubscribe. Equality-compared by identity —
    don't rely on its internals."""
    kind: str
    fn: object


class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Subscription]] = {}

    def on(self, kind: str, fn) -> Subscription:
        sub = Subscription(kind=str(kind), fn=fn)
        self._subs.setdefault(sub.kind, []).append(sub)
        return sub

    def off(self, sub: Subscription) -> bool:
        bucket = self._subs.get(sub.kind)
        if not bucket:
            return False
        try:
            bucket.remove(sub)
            return True
        except ValueError:
            return False

    def emit(self, kind: str, payload=None) -> None:
        # Snapshot the bucket before iterating — handlers are allowed to
        # subscribe or unsubscribe during dispatch.
        for sub in list(self._subs.get(str(kind), ())):
            try:
                if payload is None:
                    sub.fn()
                elif isinstance(payload, tuple):
                    sub.fn(*payload)
                else:
                    sub.fn(payload)
            except Exception as exc:
                print(f'event handler for {kind!r} failed: {exc}')
