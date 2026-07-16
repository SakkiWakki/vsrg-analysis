"""fluXis storyboard-script recorder (a Mode-2 port: run the original
chart code once, record what it builds).

Scripts are referenced by `.fsb` Script elements (type 3, parameter
`path`). Execution model, mirrored from StoryboardScriptRunner + the
install's typed Lua metas (scripting/library/*.lua):

- top-level chunk runs once, then `process(parent)` is called once
  per Script element, with `parent:param(key, fallback)` reading that
  element's parameters (DefineParameter fallbacks apply first).
- element constructors (StoryboardBox/Sprite/Text/...) build tables;
  `el:animate(type, time, len, startVal, endVal, ease)` appends
  keyframes; `Add(el)` submits. Animation times are authored ABSOLUTE
  under version 1 (the default; fluXis's Add migrates by subtracting
  element start) and element-relative after `SetVersion(2)`. The
  recorder normalizes everything to absolute, so recorded elements
  compile through the `.fsb` compiler with version-1 semantics.
- enum helpers (Easing/Anchor/Layer/BlendMode/HitObjectType) convert
  names to the engine's numeric values; `map:*InRange` query the
  parsed chart; AudioAnalyzer serves precomputed FFT frames
  (script_fft, approximate); skin metrics return fixed defaults for
  fluXis's stock skin since we don't load fluXis skins.
- `RandomRange` is seeded per script path: fluXis rolls a live RNG,
  we keep playback and caching deterministic.

Scripts run under Lua 5.4 (fluXis embeds NLua) inside the sandbox
from analysis/player/render/lua.
"""
from __future__ import annotations

import random
from bisect import bisect_right
from pathlib import Path

from analysis.games.fluxis import script_fft

_EASINGS = (
    'None', 'Out', 'In', 'InQuad', 'OutQuad', 'InOutQuad', 'InCubic',
    'OutCubic', 'InOutCubic', 'InQuart', 'OutQuart', 'InOutQuart',
    'InQuint', 'OutQuint', 'InOutQuint', 'InSine', 'OutSine',
    'InOutSine', 'InExpo', 'OutExpo', 'InOutExpo', 'InCirc', 'OutCirc',
    'InOutCirc', 'InElastic', 'OutElastic', 'OutElasticHalf',
    'OutElasticQuarter', 'InOutElastic', 'InBack', 'OutBack',
    'InOutBack', 'InBounce', 'OutBounce', 'InOutBounce', 'OutPow10',
)
_EASING_IDS = {name: i for i, name in enumerate(_EASINGS)}

_ANCHOR_BITS_X = {'Left': 8, 'Centre': 16, 'Right': 32}
_ANCHOR_BITS_Y = {'Top': 1, 'Centre': 2, 'Bottom': 4}

_LAYERS = {'Background': 0, 'Foreground': 1, 'Overlay': 2}
_BLEND_MODES = {'None': 0, 'Inherit': 1, 'Mix': 2, 'Difference': 3,
                'Add': 4, 'Subtract': 5, 'Screen': 6, 'Multiply': 7,
                'Premultiplied': 8}
_HITOBJECT_TYPES = {'Normal': 0, 'Tick': 1, 'Landmine': 2}
_ANIM_TYPE_IDS = {'MoveX': 0, 'MoveY': 1, 'Scale': 2, 'ScaleVector': 3,
                  'Width': 4, 'Height': 5, 'Rotate': 6, 'Fade': 7,
                  'Color': 8, 'Border': 9}

_EVENT_STREAMS = {
    'BeatPulse': 'beatpulse', 'ColorFade': 'colorfade', 'Flash': 'flash',
    'LaneSwitch': 'laneswitch', 'LayerFade': 'layerfade', 'Pulse': 'pulse',
    'Shader': 'shader', 'Shake': 'shake', 'HitObjectEase': 'hitease',
    'ScrollMultiplier': 'scroll-multiply', 'TimeOffset': 'time-offset',
    'PlayfieldMove': 'playfieldmove', 'PlayfieldRotate': 'playfieldrotate',
    'PlayfieldScale': 'playfieldscale', 'Loop': 'loops',
    'CameraMove': 'camera-move', 'CameraRotate': 'camera-rotate',
    'CameraScale': 'camera-scale',
}

# Stock-skin stand-ins; fluXis reads these from the active skin json.
_SKIN_COLUMN_WIDTH = 114.0
_SKIN_HIT_POSITION = 130.0

_ELEMENT_BOOTSTRAP = """
local function base(type_id)
    local el = {
        __type = type_id, __anims = {},
        layer = 0, z = 0, time = 0, endtime = 0,
        anchor = 9, origin = 9, x = 0, y = 0,
        blend = false, blendMode = 4,
        width = 0, height = 0, color = 0xFFFFFFFF,
    }
    function el:animate(atype, time, len, startVal, endVal, ease)
        self.__anims[#self.__anims + 1] = {
            type = atype, time = time, len = len,
            startVal = tostring(startVal), endVal = tostring(endVal),
            ease = ease or 0,
        }
    end
    function el:param(key, fallback)
        return fallback
    end
    return el
end
function StoryboardBox() return base(0) end
function StoryboardSprite()
    local el = base(1); el.texture = ''; return el
end
function StoryboardText()
    local el = base(2); el.text = ''; el.size = 20; return el
end
function StoryboardCircle() return base(4) end
function StoryboardOutlineCircle()
    local el = base(5); el.border = 4; return el
end
function StoryboardSkinSprite(spr)
    local el = base(6); el.sprite = spr or 0
    el.lane = 1; el.keycount = 4
    return el
end
function StoryboardOutlineBox()
    local el = base(7); el.border = 4; return el
end
"""


def _anchor_value(name: str) -> int:
    name = str(name)
    if name == 'Centre':
        return _ANCHOR_BITS_X['Centre'] | _ANCHOR_BITS_Y['Centre']
    for y_name, y_bit in _ANCHOR_BITS_Y.items():
        if name.startswith(y_name):
            x_bit = _ANCHOR_BITS_X.get(name[len(y_name):], 0)
            if x_bit:
                return x_bit | y_bit
    return _ANCHOR_BITS_X['Left'] | _ANCHOR_BITS_Y['Top']


def _easing_id(value) -> int:
    if isinstance(value, str):
        return _EASING_IDS.get(value, 0)
    return int(value or 0)


def _pack_color(value) -> int:
    if value is None:
        return 0xFFFFFFFF
    if isinstance(value, (int, float)):
        return int(value) & 0xFFFFFFFF

    def byte(key):
        channel = value[key]
        return max(0, min(255, round(float(channel or 0.0) * 255)))
    return (byte('r') << 24) | (byte('g') << 16) | (byte('b') << 8) \
        | byte('a')


class ScriptRecorder:
    """One script file's sandbox + the elements it Add()s."""

    def __init__(self, chart: dict, resolution: tuple, assets_dir,
                 audio_path=None, seed: str = ''):
        from analysis.player.render.lua import LuaHost

        self.recorded: list = []
        self._chart = chart
        self._audio_path = audio_path
        self._version = 1
        self._param_defaults: dict = {}
        self._rng = random.Random(f'fluxis-script:{seed}')
        self._timing = sorted(
            (float(t), float(bpm))
            for t, bpm in chart.get('timing_points') or [])

        self._host = LuaHost(dialect='lua54')
        self._expose_api(resolution)
        self._host.run(_ELEMENT_BOOTSTRAP, name='element-bootstrap')

    # -- script lifecycle -------------------------------------------------

    def run(self, source: str, name: str = 'storyboard-script') -> None:
        self._host.run(source, name=name)

    def process(self, script_element: dict) -> None:
        """Invoke the script's process(parent) for one Script element."""
        parameters = script_element.get('parameters') or {}
        parent = self._host.to_lua({
            key: script_element.get(key, 0)
            for key in ('layer', 'time', 'endtime', 'anchor', 'origin',
                        'x', 'y', 'width', 'height', 'color')
        })

        def param(_self, key, fallback=None):
            if key in parameters:
                return self._host.to_lua(parameters[key])
            if key in self._param_defaults:
                return self._host.to_lua(self._param_defaults[key])
            return fallback

        parent['param'] = param
        self._host.call('process', parent)

    # -- API --------------------------------------------------------------

    def _expose_api(self, resolution) -> None:
        host = self._host
        host.expose('Add', lambda el: self._add(el))
        host.expose('SetVersion', self._set_version)
        host.expose('RandomRange', lambda a, b: self._rng.uniform(
            float(a), float(b)))
        host.expose('BPMAtTime', self._bpm_at)
        host.expose('DefineParameter', self._define_parameter)
        host.expose('print', lambda text: print(f'[fluxis script] {text}'))

        host.expose('Easing', lambda name: _EASING_IDS.get(str(name), 0))
        host.expose('Anchor', lambda name: _anchor_value(name))
        host.expose('Layer', lambda name: _LAYERS.get(str(name), 0))
        host.expose('BlendMode', lambda name: _BLEND_MODES.get(str(name), 2))
        host.expose('HitObjectType',
                    lambda name: _HITOBJECT_TYPES.get(str(name), 0))
        host.expose('Color4', self._color4)
        host.expose('Vector2', lambda x, y: host.to_lua(
            {'x': float(x), 'y': float(y)}))

        host.expose('screen', {'x': float(resolution[0]),
                               'y': float(resolution[1])})
        meta = self._chart.get('meta') or {}
        host.expose('metadata', {
            'title': meta.get('title', ''), 'artist': meta.get('artist', ''),
            'mapper': meta.get('mapper', ''),
            'difficulty': meta.get('difficulty', ''),
            'background': meta.get('background', ''),
            'cover': meta.get('cover', ''),
        })
        host.expose('settings', {'scrollspeed': 25.0, 'upscroll': False})
        host.expose('skin', {
            'sprratio': lambda _s, _name=None: 1.0,
            'colwidth': lambda _s, _mode=None: _SKIN_COLUMN_WIDTH,
            'hitpos': lambda _s, _mode=None: _SKIN_HIT_POSITION,
            'recoffset': lambda _s, _mode=None: 0.0,
            'recfirst': lambda _s, _mode=None: 0.0,
        })
        host.expose('map', {
            'NotesInRange': self._notes_in_range,
            'TimingPointsInRange': self._timing_in_range,
            'ScrollVelocitiesInRange': self._svs_in_range,
            'HitSoundFadesInRange': lambda _s, *_a: host.to_lua([]),
            'EventsInRange': self._events_in_range,
        })
        host.expose('AudioAnalyzer',
                    {'AmplitudesInRange': self._amplitudes_in_range})
        host.expose('FFTParameters', script_fft.PRESETS)

    def _set_version(self, version) -> None:
        self._version = int(version)

    def _define_parameter(self, key, _title, _kind, fallback=None) -> None:
        self._param_defaults[str(key)] = fallback

    def _bpm_at(self, time_ms) -> float:
        if not self._timing:
            return 120.0
        idx = bisect_right([t for t, _ in self._timing], float(time_ms)) - 1
        return self._timing[max(0, idx)][1]

    def _color4(self, r, g, b, a=1.0):
        return self._host.to_lua({'r': float(r), 'g': float(g),
                                  'b': float(b), 'a': float(a)})

    # -- map queries --------------------------------------------------------

    def _notes_in_range(self, _self, start, end, kind=None):
        if isinstance(kind, str):
            kind = _HITOBJECT_TYPES.get(kind, 0)
        notes = []
        for h in self._chart.get('hitobjects') or []:
            if not float(start) <= h['time'] <= float(end):
                continue
            if kind is not None and int(h.get('type', 0)) != int(kind):
                continue
            hold = (h['end_time'] - h['time']) if h.get('end_time') else 0.0
            notes.append({
                'time': h['time'], 'lane': h['column'] + 1,
                'visualLane': h['column'] + 1, 'holdTime': hold,
                'hitSound': '', 'group': '', 'hidden': False,
                'type': int(h.get('type', 0)),
            })
        return self._host.to_lua(notes)

    def _timing_in_range(self, _self, start, end):
        return self._host.to_lua(
            [{'time': t, 'bpm': bpm} for t, bpm in self._timing
             if float(start) <= t <= float(end)])

    def _svs_in_range(self, _self, start, end):
        return self._host.to_lua(
            [{'time': float(t), 'multiplier': float(m)}
             for t, m in self._chart.get('scroll_velocities') or []
             if float(start) <= float(t) <= float(end)])

    def _events_in_range(self, _self, start, end, kind):
        stream = self._chart.get('effect_streams', {}).get(
            _EVENT_STREAMS.get(str(kind), ''), [])
        return self._host.to_lua(
            [e for e in stream if isinstance(e, dict)
             and float(start) <= float(e.get('time', 0.0)) <= float(end)])

    def _amplitudes_in_range(self, _self, start, end, interval,
                             count=None, params=None):
        frames = script_fft.amplitudes_in_range(
            self._audio_path, float(start), float(end), float(interval),
            count=int(count or 256),
            params=dict(params.items()) if params is not None else None)
        return self._host.to_lua([self._fft_frame(f) for f in frames])

    def _fft_frame(self, frame: dict) -> dict:
        amplitudes = frame['amplitudes']
        peak = max(amplitudes, default=0.0)
        average = (sum(amplitudes) / len(amplitudes)) if amplitudes else 0.0
        bands = {'low': frame['low'], 'mid': frame['mid'],
                 'high': frame['high'], 'total': frame['total'],
                 'GetDominantBand': lambda _s: float(
                     max(range(3), key=lambda i: (frame['low'], frame['mid'],
                                                  frame['high'])[i]))}
        return {
            'amplitudes': amplitudes,
            'bands': bands,
            'IsSilent': lambda _s, threshold=0.001: peak <= float(threshold),
            'DetectBeat': lambda _s, threshold=0.5: frame['low'] >= float(
                threshold),
            'GetPeakAmplitude': lambda _s: peak,
            'GetAverageAmplitude': lambda _s: average,
            'GetPeakFrequencyBin': lambda _s: float(
                amplitudes.index(peak) if amplitudes else 0),
        }

    # -- element recording --------------------------------------------------

    def _add(self, lua_el) -> None:
        el_start = float(lua_el['time'] or 0.0)
        raw = {
            'type': int(lua_el['__type']),
            'layer': int(lua_el['layer'] or 0),
            'z-index': int(lua_el['z'] or 0),
            'start': el_start,
            'end': float(lua_el['endtime'] or 0.0),
            'anchor': int(lua_el['anchor'] or 9),
            'origin': int(lua_el['origin'] or 9),
            'x': float(lua_el['x'] or 0.0),
            'y': float(lua_el['y'] or 0.0),
            'width': float(lua_el['width'] or 0.0),
            'height': float(lua_el['height'] or 0.0),
            'blend': bool(lua_el['blend']),
            'color': _pack_color(lua_el['color']),
            'parameters': self._element_parameters(lua_el),
            'animations': self._element_animations(lua_el, el_start),
        }
        self.recorded.append(raw)

    def _element_parameters(self, lua_el) -> dict:
        match int(lua_el['__type']):
            case 1:
                return {'file': str(lua_el['texture'] or '')}
            case 2:
                return {'text': str(lua_el['text'] or ''),
                        'size': float(lua_el['size'] or 20.0)}
            case 5 | 7:
                return {'border': float(lua_el['border'] or 4.0)}
            case 6:
                return {'sprite': lua_el['sprite'],
                        'lane': int(lua_el['lane'] or 1),
                        'keycount': int(lua_el['keycount'] or 4)}
        return {}

    def _element_animations(self, lua_el, el_start: float) -> list:
        # Normalize to absolute times: version 1 scripts author
        # absolute already; version 2+ author element-relative.
        offset = el_start if self._version >= 2 else 0.0
        anims = lua_el['__anims']
        out = []
        for i in range(1, len(anims) + 1):
            anim = anims[i]
            type_id = _ANIM_TYPE_IDS.get(str(anim['type']))
            if type_id is None:
                print(f"fluxis script: unknown animation type"
                      f" {anim['type']!r}; skipped")
                continue
            out.append({
                'start': float(anim['time'] or 0.0) + offset,
                'duration': float(anim['len'] or 0.0),
                'easing': _easing_id(anim['ease']),
                'type': type_id,
                'start-value': str(anim['startVal']),
                'end-value': str(anim['endVal']),
                'use-start': True,
            })
        return out


def record_script_elements(fsb_raw: dict, fsb_dir, chart: dict,
                           audio_path=None) -> list:
    """Run every Script element's file and return the recorded raw
    elements (`.fsb` schema, absolute animation times: compile with
    version-1 semantics). Script problems skip that script with a
    warning; storyboards must never break chart loading."""
    scripts = [e for e in fsb_raw.get('elements') or []
               if isinstance(e, dict) and int(e.get('type', -1)) == 3]
    if not scripts:
        return []

    resolution = fsb_raw.get('resolution') or {}
    size = (float(resolution.get('x', 1920.0) or 1920.0),
            float(resolution.get('y', 1080.0) or 1080.0))

    by_path: dict = {}
    for element in scripts:
        path = str((element.get('parameters') or {}).get('path', '')).strip()
        if path:
            by_path.setdefault(path, []).append(element)

    recorded = []
    for rel_path, elements in by_path.items():
        script_file = Path(fsb_dir) / rel_path
        try:
            source = script_file.read_text(encoding='utf-8')
        except OSError:
            print(f'fluxis script missing: {script_file}')
            continue
        try:
            recorder = ScriptRecorder(chart, size, fsb_dir,
                                      audio_path=audio_path, seed=rel_path)
            recorder.run(source, name=rel_path)
            for element in elements:
                recorder.process(element)
        except Exception as exc:
            print(f'fluxis script {rel_path!r} failed: {exc}')
            continue
        recorded.extend(recorder.recorded)
    return recorded
