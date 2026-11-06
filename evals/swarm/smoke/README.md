# Swarm smoke task set

Six small bundles in the format of `docs/SWARM-EVAL-DESIGN.md` §2, used to smoke-test a
swarm run (Mac small model, or the rented GPU) before the full task set exists.

| id | kind | units | what the units are |
|---|---|---|---|
| `textkit-3` | fanout | 3 | independent utility modules: slugify, word wrap, roman numerals |
| `kvstores-4` | fanout | 4 | adapters behind a given `KVStore` protocol: memory, JSON file, sqlite, TTL wrapper |
| `minicli-3` | fanout | 3 | argparse subcommands plugged into a given dispatcher: stats, convert, hashsum |
| `formats-2` | fanout | 2 | duration parser/formatter, strict INI parser/dumper |
| `validators-4` | fanout | 4 | Luhn, ISBN, IBAN, semver behind a given `validate()` dispatcher |
| `control-calc` | control | 1 | expression language: lexer -> parser -> evaluator, a strict dependency chain with internal interfaces left to the implementer (not decomposable) |

Fan-out units own disjoint files and only share a given, read-only interface, so 2-4
agents can work on them in parallel without coordination.

## Bundle layout

```
<id>/task.toml   id, kind (fanout|control), spec, units[{id, owns[paths], hidden_test}], budget
<id>/repo/       starting tree: pure-stdlib Python, stubs + a few visible tests (tests/)
<id>/hidden/     one test file per unit + test_integration.py; copied in only at grading
<id>/solution/   reference answer, an overlay on repo/ (oracle arm and validation only)
```

Hidden tests use cases that are not in the visible tests. Grading copies `hidden/` to
`hidden_tests/` next to the agent's tree and runs pytest there.

## Validation (oracle / null)

```bash
uv run python evals/swarm/validate_bundles.py            # all of evals/swarm/smoke
uv run python evals/swarm/validate_bundles.py <bundle>   # one bundle; --json for machines
```

For each bundle, in fresh temp copies and in subprocesses with a timeout and sockets
disabled, the validator checks the schema, that **repo/ + solution/ passes every hidden
test** (and the visible tests), and that **untouched repo/ fails every unit's hidden test
file** and the integration test. `tests/test_swarm_smoke_bundles.py` runs it in CI and
also checks that it catches a broken oracle and a unit that "passes" with no work.

`ruff.toml` here relaxes a few import-style rules for bundle sources; they are task
material, not memfun code (pyright already excludes `evals/`).
