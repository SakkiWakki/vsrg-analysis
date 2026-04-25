# Security model

## Goal

The plugin system is designed so that a third-party plugin ; including one
submitted as a contribution to `plugins/builtin/` ; cannot harm users who
install it. "Harm" means: reading another user's files, exfiltrating data over
the network, reading another plugin's private config, or corrupting shared
application state.

This is a **best-effort** model, not a formal security guarantee. The
mitigations are layered so that bypassing one layer still requires bypassing
others. The threat model is a lazy or opportunistic attacker ; someone who
adds `import os` to a plugin hoping it goes unnoticed, not someone who has
already compromised the repository and can modify the runtime.

## Trust levels

| Location | Trust level | Rationale |
|---|---|---|
| `plugins/unsafe/<bundle>/` | **Trusted** | Explicit user opt-in. The name signals the user's choice. Full Python access. |
| `plugins/builtin/<bundle>/` | **Sandboxed** | Ships with the app but treated as untrusted. An attacker could submit a malicious PR. The sandbox and verifier enforce the same rules as any third-party plugin. |
| `plugins/<bundle>/` (user-installed) | **Sandboxed** | Third-party. Same rules as builtins. |
| `$EA_PLUGINS_PATH` bundles | **Sandboxed** | User-configured extra paths. Same rules. |
| `~/.config/vsrg-analysis/plugins/` | **Sandboxed** | Per-user install dir. Same rules. |

Note: `plugins/builtin/` being sandboxed means it is held to the *same*
standard as third-party plugins. It is not trusted by virtue of being in the
repository. This is intentional ; it removes the assumption "if it shipped
with the app it must be safe" and ensures the enforce machinery actually runs
on all non-`unsafe/` code.

## Enforcement layers

### 1. Import allow-list + targeted deny-list

Sandboxed modules run with a patched `__import__` that rejects any module
not on the explicit allow-list (`_HOST_API_ALLOW | _STDLIB_ALLOW | _THIRDPARTY_ALLOW`
in `analysis/plugins/sandbox.py`). The explicit deny-list (`_EXPLICIT_DENY`) is
checked first and beats the allow-list ; this closes submodule escape vectors
even when a parent package is allowed:

- **NumPy** is allowed but `numpy.ctypeslib`, `numpy.ctypes`, `numpy.distutils`,
  `numpy.f2py`, and `numpy.testing` are explicitly denied. `numpy.ctypeslib`
  can load arbitrary shared libraries; `numpy.ctypes` is a module-level alias
  to `ctypes`. The raw `__array_interface__` pointer is harmless when `ctypes`
  is unreachable.
- **Frame walking** ; `gc`, `inspect`, `traceback`, and `linecache` are
  explicitly denied. `gc.get_objects()` walks to live frame objects; `inspect`
  and `traceback` expose `sys._getframe()` equivalents without importing `sys`.
- **matplotlib** is allowed but `matplotlib.testing` is denied (runs subprocess
  comparisons).

### 2. Restricted builtins + traceback stripping

The `__builtins__` dict in sandboxed modules has dangerous names removed:
`open`, `eval`, `exec`, `compile`, `input`, `memoryview`, `globals`, `locals`,
`vars`. All names starting with `_` (except `__build_class__`, `__name__`,
`__doc__`) are also removed. `__import__` is replaced with the gated version.

Additionally, exceptions that escape sandboxed module execution have their
tracebacks stripped (`exc.with_traceback(None)`) before being returned to the
loader. Without this, a plugin could catch an exception and walk
`exc.__traceback__.tb_frame.f_globals` to recover the blocked builtins dict
with no imports required.

### 3. Symbolic verifier (Z3-assisted)

Before a sandboxed module's source is executed, its AST is analysed by the
verifier (`analysis/plugins/verifier/`). The verifier:

- Builds a call graph from plugin entry points (`draw`, `register_components`,
  `register_sidebar`, etc.) and traces all reachable call sites including
  transitive intra-module calls.
- Checks every reachable call against a set of static sinks: blocked names
  (`open`, `eval`, `setattr`, `getattr`, `object.__setattr__`, etc.) and
  blocked attribute accesses (`__class__`, `__dict__`, `__builtins__`, etc.).
- Uses Z3 string reasoning to verify that `config.set(field, ...)` calls
  are always scoped to the plugin's own namespace ; detecting path traversal
  even when the field is a non-literal variable.

Verification failure is a **hard block**: the module is not loaded and the
error surfaces in the Plugins panel.

### 4. Cross-plugin isolation (interaction bus)

Plugins interact with each other only through a typed interaction bus
(`analysis/components/interaction.py`). A plugin must explicitly declare what
it exposes (`exposes=`) and what it subscribes to (`subscribes=`) in its
manifest. The host only delivers data that the source plugin declared as
exposed. Plugins with no `exposes` declaration are invisible to peers.

Event payloads are sanitized to primitive types (str, int) with byte-length
limits, reducing the exfiltration surface even if an attacker controls a
plugin that emits events.

### 5. Config isolation

Plugin config is scoped to `plugins.<escaped_key>.settings.*`. A plugin
cannot read or write another plugin's settings through `PluginConfig`. The
verifier additionally checks that `config.set()` field arguments are always
within the plugin's own namespace subtree.

## What the model does NOT protect against
- **Repository compromise.** If an attacker can merge arbitrary code into
  the repo and cut a release, none of these mitigations help. The model
  assumes the release build process is not compromised.
- **`plugins/unsafe/` contents.** Code in `unsafe/` has full Python access
  by design. Users who install bundles from `unsafe/` or configure
  `$EA_PLUGINS_PATH` to point at untrusted code do so at their own risk.

## Future directions

- Z3 verification of cross-plugin bus payload shapes (currently checked
  only at runtime).
- Tightening numpy by removing `ctypeslib` from the effective surface.
- Moving `plugins/unsafe/osu_live/` to `analysis/games/osu/` as trusted
  host infrastructure so the plugin directory contains only verifiable code.
- Symbolic execution of more complex invariants as they are identified.
