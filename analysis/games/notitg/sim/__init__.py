"""Headless SM engine simulation for the NotITG modfile compiler.

See DESIGN_engine_loop.md. `actor` is the simulated-and-recorded Actor;
env/loop/record/producers land in later phases.
"""
from analysis.games.notitg.sim.actor import OscSpan, SimActor

__all__ = ['OscSpan', 'SimActor']
