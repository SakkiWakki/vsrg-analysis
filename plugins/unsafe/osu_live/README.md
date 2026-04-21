# osu! Live

Live visualizations of the currently-playing osu! mania session,
reading directly from the game's memory.

## Two data sources

The `OsuLiveClient` poller picks the first source that works per tick:

1. **Native reader** (preferred). A small Rust PyO3 extension
   (`analysis/games/osu/native/`) that scans osu!'s memory via
   `process_vm_readv` on Linux/wine and reads the gameplay struct
   directly. ~40 µs per read, no server, no port.
2. **HTTP fallback** ([tosu](https://github.com/tosuapp/tosu)'s
   `/json/v2`). Used only if the native extension isn't built or
   osu!'s binary updated ahead of our signatures and the pattern
   scan fails.

`client.py` normalizes both sources into the same tosu-shaped dict
before building a `LiveSnapshot`, so viz panels don't know or care
which source produced the frame.

## Building the native reader

Requires a Rust toolchain (`cargo`, `rustc`) and `maturin`.

```sh
pip install maturin
cd analysis/games/osu/native
maturin develop
```

That drops `osu_memory_native.so` into the active venv. The client
imports it lazily; if the import fails the HTTP fallback is used.

Mania-only in v1. The pointer chain and struct offsets are derived
from tosu (see `analysis/games/osu/native/src/signatures.rs` for
provenance); the
chain is walked in exactly one place (`reader.rs::gameplay_pointers`)
so adding other modes later is additive.

## Requirements

- **Native path**: osu! running under wine on Linux. Tested with the
  wine prefix at `~/.local/share/osuconfig/wine-osu`.
- **HTTP fallback**: tosu running locally (default
  `http://127.0.0.1:24050`). See upstream docs.

This is an `unsafe/` bundle — it reads another process's memory and
makes outbound HTTP calls, which sandboxed plugins cannot. The
read-only memory access is scoped to the osu! pid and same-uid via
`process_vm_readv` (no ptrace, no injection).

## Current state

v1 proof-of-concept. Only `Live: drift (hands × time)` is wired; more
follow the same ~20-line wrap pattern once the first one is validated
on a real session.

Known limits:

- Only hits are surfaced, no per-hit column. `columns` round-robin
  across lanes so hand-split viz have *some* signal, but it is not
  true lane data.
- `noterows` is synthesized as a monotonic counter.
- Native reader doesn't yet expose UR directly (computed lazily from
  hit errors when the viz needs it).

## Signature maintenance

When osu! updates and the pattern scan fails, the native extension
surfaces a clear error naming the signature. Update
`analysis/games/osu/native/src/signatures.rs` and rebuild with
`maturin develop`. The
HTTP fallback keeps the plugin usable during the signature lag.
