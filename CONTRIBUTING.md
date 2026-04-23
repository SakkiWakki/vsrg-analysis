# Contributing

## Honesty about LLM usage

If you used an AI assistant (Claude, Cursor, Copilot, ChatGPT, etc.) to
generate or substantially modify code in a pull request, say so in the
PR description. Line-by-line attribution isn't required; something like
"used Claude for the refactor in X" or "scaffolded with Cursor, hand-edited
throughout" is enough.

This isn't a filter against AI. AI is fine. It's about keeping the review
signal honest so reviewers know what to scrutinize.

## Know what your code does

Before opening a PR, you should be able to, unaided, answer:

- What does each new function do, and why is it there?
- What invariants does it rely on? What does it assume about its inputs?
- What's the failure mode if those assumptions break?
- What did you consider and decide *not* to do?

If you can't answer these for a chunk of code, go back and read it until
you can, or strip it out. AI assistance is fine; submitting code you
don't personally understand is not.

## Code quality

### General principles

- **Write code that doesn't need comments.** Reach for better names,
  named constants, and small helpers before reaching for a comment.
  When a comment *is* needed, it's load-bearing (a "why", a precondition,
  an upstream reference).
- **Don't launder booleans.** If a function answers a yes/no question,
  compute the answer once from named predicates. Don't sprinkle
  `return True` / `return False` across the body.
- **Group code blocks.** `if` / `for` / `while` / `with` / `try` /
  `match` stays with the setup lines its body uses. Separate independent
  groups with a blank line.
- **Extract small helpers.** If a pattern repeats twice, pull it out.
  Lambda, method, or free function based on what fits.
- **Choose the lightest container that fits.** primitive > tuple >
  list/numpy > dict > dataclass > class. Each step up adds overhead.
- **User-exposed code is defensive; internal code is trusting.**
  Validate inputs at system boundaries (CLI args, file parsers, network,
  FFI, GUI handlers). Downstream of that boundary, trust your own code.
- **Use `match` over if/elif** when dispatching on literal values (3+
  cases, single value). Python 3.10+ is the project baseline.
- **Keep imports at module top.** Function-local `import` statements
  re-run on every call. Only use them when breaking a circular import,
  and leave a one-line comment explaining why.
- **No em-dashes** (the U+2014 character) in source files. They break
  some parsers. Use `-` or `--`. Clankers love inserting em-dashes in
  comments, make sure you don't leave them in!
- **Name magic numbers.** Even a one-line constant with a descriptive
  name beats an inline literal in a formula.
- **Keep functions small and single-purpose.** If you can't state what
  a function does in one short sentence, it's too big.
- **Refactor as a human if needed.** You may ask clankers to refactor
  for you. But you must verify that the refactoring is up to code quality.

### Style

- Python 3.10 or newer.
- Follow PEP 8 for formatting. Don't fight the project's existing
  conventions.
- Tests live under `tests/`. Run `pytest` before pushing.

## Tools

Before submitting:

- `pytest tests/` must pass.
- `pylint analysis/` - address new warnings on files you touched. Not
  zero-warnings across the whole tree; we're not there yet. The goal is
  that your PR doesn't introduce new warnings.
- `radon cc -s analysis/` - a complexity report maintainers use to
  spot refactor candidates. The strictness bar (what's "too complex")
  is a maintainer judgment call rather than a fixed numeric cutoff;
  expect pushback on anything above grade B, and on anything above
  grade A that could reasonably be simpler. If a function has to be
  complex, justify it in the PR description.

Install dev tools:

```
pip install pylint radon
```

These are developer-only tools and deliberately not in `requirements.txt`.

There is no committed pylint or radon config yet. Use defaults.
