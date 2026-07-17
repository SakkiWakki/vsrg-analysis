"""Headless SM engine simulation for the NotITG modfile compiler.

See DESIGN_engine_loop.md. `actor` is the simulated-and-recorded Actor,
`env` the engine surface, `loop` the tick driver, `record` the stream
shaping; producers land in phase 3.
"""
from analysis.games.notitg.sim.actor import OscSpan, SimActor
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.loop import (
    SimResult, run_chart_sim, run_sim)
from analysis.games.notitg.sim.record import coalesce_applied, summarize

__all__ = ['OscSpan', 'SimActor', 'SimEnvironment', 'SimResult',
           'coalesce_applied', 'run_chart_sim', 'run_sim', 'summarize']
