"""Bridge: harvested modfile windows -> compiled mod channels.

gat-style templates run a per-frame reader: `mod,clearall`, then every
entry whose window contains the current beat/time re-applies its
modstring (later table entries win overlaps). Compiled equivalent:
each window emits target=percent at its start and target=0 at its end
(the clearall revert), same approach speed both ways; overlaps resolve
by emit order at equal times. The channels module then solves the
linear approach into piecewise segments.

`compile_modfile` already resolves every window to seconds
(`t_start`/`t_end`), so `ModEvent.beat` carries seconds here and
`ModChannels.compile` runs on its identity clock.

Modstring grammar (PlayerOptions::FromString subset): comma-separated
tokens, each `[*S] [P% | P | no] name`; `no name` = 0 percent; a bare
name = 100%. `*S` = approach speed in fraction/second, absent = the
engine default 1.0, `*-1` (any S <= 0) = snap. Unknown mod names
still become channels (the pipeline ignores names it has no formula
for); engine view controls and x/C/M speed mods are dropped here.

Players: rows without a player apply to everyone -> player 0; an
explicit pn maps to pn - 1 for the future multi-field split.
"""
from __future__ import annotations

import re

from analysis.player.render.mods.channels import ModChannels, ModEvent

_TOKEN = re.compile(
    r'^(?:\*(?P<speed>-?\d+(?:\.\d+)?)\s+)?'
    r'(?:(?P<no>no)\s+|(?P<percent>-?\d+(?:\.\d+)?)%?\s+)?'
    r'(?P<name>[a-z][a-z0-9 ]*?)$')

_ENGINE_CONTROLS = {'clearall', 'overhead', 'incoming', 'space',
                    'hallway', 'distant'}
_SPEED_MOD = re.compile(r'^(?:\d+(?:\.\d+)?x|c\d+|m\d+)$')

_DEFAULT_SPEED = 1.0


def parse_modstring(modstring: str) -> list:
    """[(percent_fraction, speed, name), ...] for one ApplyGameCommand
    payload; engine controls and speed mods are dropped."""
    out = []
    for token in str(modstring).lower().split(','):
        token = ' '.join(token.split())
        if not token:
            continue
        match = _TOKEN.match(token)
        if match is None:
            continue
        name = ' '.join(match['name'].split())
        if name in _ENGINE_CONTROLS or _SPEED_MOD.match(name):
            continue
        if match['no']:
            percent = 0.0
        elif match['percent'] is not None:
            percent = float(match['percent']) / 100.0
        else:
            percent = 1.0
        speed = (float(match['speed']) if match['speed'] is not None
                 else _DEFAULT_SPEED)
        out.append((percent, speed, name.replace(' ', '')))
    return out


def compile_mod_channels(mod_events) -> ModChannels:
    """Compile `compile_modfile`'s normalized mod-window dicts
    (`t_start`/`t_end` seconds, `modstring`, `player`) into sampled
    channels."""
    events = []
    for row in mod_events:
        start = float(row['t_start'])
        end = float(row['t_end'])
        if end < start:
            continue
        raw_player = row.get('player')
        player = 0 if raw_player is None else max(0, int(raw_player) - 1)
        for percent, speed, name in parse_modstring(row['modstring']):
            events.append(ModEvent(start, percent, speed, name, player))
            events.append(ModEvent(end, 0.0, speed, name, player))
    return ModChannels.compile(events)
