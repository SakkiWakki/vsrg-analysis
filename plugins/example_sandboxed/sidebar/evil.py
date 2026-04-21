"""Deliberately tries a disallowed import. The bundle loader should
refuse this file and record it in ``bundle.load_errors`` so the Plugins
panel can flag the bundle — but the rest of the bundle still loads."""
import os  # noqa: F401 — sandbox should reject this import


def register_sidebar(add):
    pass  # never reached
