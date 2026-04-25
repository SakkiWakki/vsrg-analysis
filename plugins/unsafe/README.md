# `unsafe/` ; trusted bundles that skip the sandbox

Bundles placed here run with **full Python access**: raw filesystem,
network, subprocess, threads, ctypes, anything the host process can do.
They are trusted in exactly the same way as `plugins/builtin/`.

This is an opt-in escape hatch. Use it when:

- You're prototyping a plugin that legitimately needs capabilities the
  sandbox doesn't expose (e.g. fetching data over the network, spawning
  a helper process, writing a cache file).
- You're developing a plugin locally that will eventually be promoted
  into `builtin/`, and you don't want to fight the allow-list while
  iterating.
- You're scripting something for personal use and accept the risk.

Do **not** use it when:

- You plan to distribute the bundle to other users. Ship a sandboxed
  bundle instead, or get the capability added to the host API.
- The work fits inside the sandbox's allow-list. There's no reason to
  reach for `unsafe/` if `math` + `numpy` + the host API already cover
  what you need.

## Layout

Identical to sandboxed bundles ; see [`../README.md`](../README.md).
Drop each bundle in its own subdirectory:

```
plugins/unsafe/
  my_experiment/
    manifest.toml
    sidebar/
    viz/
    replay/
    overlay/
```

## Security

There is no sandboxing applied to anything under this directory. A
malicious bundle placed here can read your files, make network
connections, install or overwrite software, etc. Treat installing an
`unsafe/` bundle the same way you'd treat running a random script from
the internet ; only install from sources you trust.

## Deprecation notice

**`unsafe/` is a temporary escape hatch.** Once the plugin ecosystem
matures and the host API surface is wide enough to cover realistic
plugin needs, this directory will likely be removed. The intended
long-term shape is: every non-built-in plugin runs sandboxed, and any
capability the sandbox doesn't expose either gets a proper host-API
surface or stays out of plugin-land entirely.

If you're relying on an `unsafe/` bundle, expect to either:

- port it to the sandbox once the host API covers what it needs, or
- upstream it into `builtin/` if it's broadly useful.

Don't build a production workflow around assuming `unsafe/` will always
exist.
