/**
 * FakeWebSocket -- injected at DocumentCreation time via QWebEngineScript,
 * before any page JS runs (including ReconnectingWebSocket and socket.js).
 *
 * Problems solved:
 *
 * 1. ReconnectingWebSocket reads WebSocket.CONNECTING/OPEN/etc. at parse
 *    time. We must be installed first.
 *
 * 2. WebSocketManager passes window.location.host as the WS host, which is
 *    empty for file:// pages. We patch window.location so the host reads as
 *    a fake local address ("localhost:24050"), making all URL constructions
 *    valid. The shim then intercepts the connection regardless of URL.
 *
 * 3. Overlays open a second socket to /websocket/commands for settings.
 *    We stub those too -- commands return an empty response so the overlay
 *    doesn't crash, and filter sends on the commands socket are ignored.
 *
 * 4. /websocket/v2/precise -- some overlays open a separate precise socket
 *    for per-hit error streaming. The shim intercepts it; Python pushes
 *    hit-error state via the same _tosuPush channel, tagged by URL suffix.
 *
 * Protocol:
 *   JS -> Python  ws.send(`applyFilters:${JSON.stringify(filters)}`)
 *   Python -> JS  window._tosuPush(jsonString)   [main + precise]
 *              or window._tosuPushPrecise(jsonString)  [precise only]
 */
(function () {
    'use strict';

    // ------------------------------------------------------------------ //
    // window.location.host is empty under file:// and non-configurable in
    // Chromium, so we can't patch it directly. Instead we patch the URL
    // constructor: whenever it sees a tosu-style URL with an empty host
    // (ws://:/... or ws://undefined/...), we rewrite to a well-formed
    // localhost variant before handing it to the native URL parser. This
    // unblocks overlays that do:
    //     new URL(`${window.location.protocol}//${window.location.host}/tokens`)
    // and then feed the .href to WebSocket.
    // ------------------------------------------------------------------ //
    var FAKE_HOST = 'localhost:24050';
    var _NativeURL = window.URL;
    var _BROKEN = /(ws|wss|http|https):\/\/(:|undefined:|undefined\/|\/)/;
    function _fixUrl(u) {
        if (typeof u !== 'string') { return u; }
        if (_BROKEN.test(u)) {
            return u.replace(
                /^(ws|wss|http|https):\/\/(?:undefined)?:?(?:undefined)?\/?/,
                '$1://' + FAKE_HOST + '/');
        }
        return u;
    }
    function ShimURL(url, base) {
        return base !== undefined
            ? new _NativeURL(_fixUrl(url), base)
            : new _NativeURL(_fixUrl(url));
    }
    ShimURL.prototype = _NativeURL.prototype;
    // Preserve static methods (createObjectURL, revokeObjectURL, parse).
    for (var k in _NativeURL) {
        if (_NativeURL.hasOwnProperty(k)) {
            try { ShimURL[k] = _NativeURL[k]; } catch (e) {}
        }
    }
    try { window.URL = ShimURL; } catch (e) {}

    // ------------------------------------------------------------------ //
    // Internal state
    // ------------------------------------------------------------------ //

    var _channel = null;
    var _queue = [];
    // Separate instance lists per endpoint so we can push differently.
    var _main_instances = [];      // /ws, /websocket/v2
    var _precise_instances = [];   // /websocket/v2/precise
    var _command_instances = [];   // /websocket/commands

    function _flushQueue() {
        var q = _queue;
        _queue = [];
        for (var i = 0; i < q.length; i++) { _sendToChannel(q[i]); }
    }

    function _sendToChannel(data) {
        if (_channel && _channel.objects && _channel.objects.bridge) {
            _channel.objects.bridge.receiveFromJs(data);
        } else {
            _queue.push(data);
        }
    }

    function _initChannel() {
        if (typeof qt === 'undefined' || !qt.webChannelTransport) {
            setTimeout(_initChannel, 10);
            return;
        }
        new QWebChannel(qt.webChannelTransport, function (ch) {
            _channel = ch;
            _flushQueue();
        });
    }
    _initChannel();

    // ------------------------------------------------------------------ //
    // FakeWebSocket
    // ------------------------------------------------------------------ //

    function _endpointOf(url) {
        if (typeof url !== 'string') return 'main';
        if (url.indexOf('/websocket/v2/precise') !== -1) return 'precise';
        if (url.indexOf('/websocket/commands')   !== -1) return 'commands';
        return 'main';
    }

    function FakeWebSocket(url) {
        this.url = String(url || '');
        this._endpoint = _endpointOf(this.url);
        this.readyState = FakeWebSocket.CONNECTING;
        this.onopen    = null;
        this.onclose   = null;
        this.onmessage = null;
        this.onerror   = null;
        this._listeners = [];
        this.bufferedAmount = 0;
        this.extensions = '';
        this.protocol   = '';
        this.binaryType = 'blob';

        switch (this._endpoint) {
            case 'precise':  _precise_instances.push(this);  break;
            case 'commands': _command_instances.push(this);  break;
            default:         _main_instances.push(this);     break;
        }

        var self = this;
        setTimeout(function () {
            self.readyState = FakeWebSocket.OPEN;
            var ev = { type: 'open', target: self };
            if (typeof self.onopen === 'function') { self.onopen(ev); }
            self._dispatch('open', ev);

            // Commands socket: send an empty settings response so overlays
            // don't crash waiting for getSettings.
            if (self._endpoint === 'commands') {
                setTimeout(function () {
                    self._push(JSON.stringify({
                        command: 'getSettings',
                        message: {}
                    }));
                }, 50);
            }
        }, 0);
    }

    FakeWebSocket.CONNECTING = 0;
    FakeWebSocket.OPEN       = 1;
    FakeWebSocket.CLOSING    = 2;
    FakeWebSocket.CLOSED     = 3;

    FakeWebSocket.prototype.send = function (data) {
        if (typeof data !== 'string') { return; }
        // Commands socket: handle sendCommand -> reply with empty getSettings
        if (this._endpoint === 'commands') {
            try {
                var msg = JSON.parse(data);
                if (msg.command === 'getSettings') {
                    var self = this;
                    setTimeout(function () {
                        self._push(JSON.stringify({ command: 'getSettings', message: {} }));
                    }, 10);
                }
            } catch (e) {}
            return;
        }
        // Main / precise: forward applyFilters to Python.
        _sendToChannel(data);
    };

    FakeWebSocket.prototype.close = function () {
        this.readyState = FakeWebSocket.CLOSED;
        var ev = { type: 'close', target: this, code: 1000, reason: '', wasClean: true };
        if (typeof this.onclose === 'function') { this.onclose(ev); }
        this._dispatch('close', ev);
        this._remove();
    };

    FakeWebSocket.prototype.addEventListener = function (type, fn) {
        this._listeners.push({ type: type, fn: fn });
    };

    FakeWebSocket.prototype.removeEventListener = function (type, fn) {
        this._listeners = this._listeners.filter(function (l) {
            return !(l.type === type && l.fn === fn);
        });
    };

    FakeWebSocket.prototype._dispatch = function (type, ev) {
        for (var i = 0; i < this._listeners.length; i++) {
            if (this._listeners[i].type === type) { this._listeners[i].fn(ev); }
        }
    };

    FakeWebSocket.prototype._push = function (jsonString) {
        var ev = new MessageEvent('message', { data: jsonString });
        if (typeof this.onmessage === 'function') { this.onmessage(ev); }
        this._dispatch('message', ev);
    };

    FakeWebSocket.prototype._remove = function () {
        var lists = [_main_instances, _precise_instances, _command_instances];
        for (var i = 0; i < lists.length; i++) {
            var idx = lists[i].indexOf(this);
            if (idx !== -1) { lists[i].splice(idx, 1); }
        }
    };

    // ------------------------------------------------------------------ //
    // Install
    // ------------------------------------------------------------------ //

    window.WebSocket = FakeWebSocket;

    // Push helpers callable from Python.
    // _tosuPush     -> main + precise instances (v2 data).
    // _tosuPushPrecise -> precise instances only (hitErrors stream).
    window._tosuPush = function (jsonString) {
        var all = _main_instances.concat(_precise_instances);
        for (var i = 0; i < all.length; i++) { all[i]._push(jsonString); }
    };
    window._tosuPushPrecise = function (jsonString) {
        for (var i = 0; i < _precise_instances.length; i++) {
            _precise_instances[i]._push(jsonString);
        }
    };

    // Expose for debugging.
    window._tosuShim = {
        main: _main_instances,
        precise: _precise_instances,
        commands: _command_instances,
    };
}());
