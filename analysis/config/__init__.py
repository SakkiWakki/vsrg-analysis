"""Process-wide config store.

Single source of truth for all app settings — paths, player defaults,
per-plugin state. See :mod:`analysis.config.store` for the API and the
schema it owns. A module-level singleton is returned by
:func:`get_config`; production code should always go through that so
every window shares the same instance and subscription graph.
"""
from analysis.config.store import ConfigStore, get_config, reset_for_tests

__all__ = ['ConfigStore', 'get_config', 'reset_for_tests']
