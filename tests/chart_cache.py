"""One NotITG compile per (chart, end) per pytest session.

A real-chart compile is the expensive thing in this suite: the load pass, then
a background sweep of the whole chart that tests have to wait for before they
can assert exact values (gat 1 ~8s, gat 2 ~14-25s). Every test that compiled
its own paid that again - and left another sweep thread running in the sim,
holding the GIL against every test that followed it.

Callers get a shallow dict copy, so a test rebinding a top-level key cannot
leak into the next one. The live objects inside (the sim, the field-instance
provider, the element tree) ARE shared, exactly as the app shares them: they
are read-through views of one finished compile.
"""
from functools import lru_cache

from analysis.games.notitg.sim.producers import (
    compile_via_sim, wait_for_upgrade)


@lru_cache(maxsize=None)
def _compiled(sm_path: str, end_seconds: float | None):
    compiled = compile_via_sim(sm_path, end_seconds=end_seconds)
    wait_for_upgrade(compiled)
    return compiled


def compiled_chart(sm_path, end_seconds: float | None = None):
    """The compiled modfile for `sm_path`, upgraded and shared session-wide."""
    result = _compiled(str(sm_path), end_seconds)
    return dict(result) if isinstance(result, dict) else result
