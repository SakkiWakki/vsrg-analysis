"""Sandboxed Lua host for chart-script recorders (Mode-2 ports).

Chart scripts are untrusted code shipped inside downloaded maps. The
host runs them once, at compile time, inside an environment holding
only a safe Lua stdlib subset plus whatever API a per-game recorder
personality exposes; scripts can never reach Python, the filesystem,
or the renderer. Personalities live with their game (e.g.
analysis/games/fluxis/lua_storyboard.py) and record what the script
builds into compiled IR.
"""
from analysis.player.render.lua.host import LuaHost

__all__ = ['LuaHost']
