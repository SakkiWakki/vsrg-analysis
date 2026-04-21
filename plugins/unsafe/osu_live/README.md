# osu! Live (tosu)

Live visualizations of the currently-playing osu!(stable|lazer) session
by consuming the [tosu](https://github.com/tosuapp/tosu) memory reader's
HTTP feed.

## Requirements

- tosu running locally (default `http://127.0.0.1:24050`).
- This is an `unsafe/` bundle — it makes outbound HTTP calls, which
  sandboxed plugins can't. If you don't trust this code, read
  `tosu_client.py`; it only does GETs against a configurable URL.

## Setup

1. Install + run tosu (see upstream docs).
2. Start a mania map. Tosu exposes gameplay state at
   `/json/v2` (we poll) or `/websocket/v2` (not used here — polling is
   simpler and fast enough for sidebar viz).
3. Open the Visualize menu in this app. Entries prefixed with "Live:"
   feed off tosu.

## Current state

v1 proof-of-concept. Only `Live: drift (hands × time)` is wired; more
to follow once the tosu→replay-dict adapter shakes out.

Known limits:

- Only hits are available (no per-note miss column), so viz that split
  hit vs miss only show hits.
- `noterows` is synthesized as a monotonic counter, not chart rows —
  viz that depend on true chart time (e.g. `scatter_timeline`) won't
  work yet.
- Rebuilding Matplotlib figures on every tick is wasteful; fine up to
  a few thousand points. Swap to `set_data` later if needed.

## Future

Once the tosu feed is well-understood we can consider writing our own
Python memory reader (skipping tosu). Not a v1 goal — tosu is the
reference until the semantics are nailed down.
