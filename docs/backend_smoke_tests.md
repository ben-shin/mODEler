# Backend Smoke Tests

mODEler includes a smoke-test runner for the recommended backend workflows.

The goal is to catch broken CLI/config/API paths before GUI development depends on them.

## Smoke test script

```text
scripts/smoke_test_backend_workflows.py
```

## Fast smoke test

Run:

```bash
python scripts/smoke_test_backend_workflows.py --stop-on-failure
```

This runs the core recommended fast workflows:

- single-species variable projection fit
- single-species variable projection multistart
- single-species variable projection model comparison
- single-species variable projection multistart model comparison
- multispecies variable projection fit
- multispecies variable projection multistart
- multispecies variable projection model comparison
- multispecies variable projection multistart model comparison

## Slow smoke test

Run:

```bash
python scripts/smoke_test_backend_workflows.py --include-slow --stop-on-failure
```

This additionally runs:

- single-species bootstrap
- single-species profile likelihood
- multispecies bootstrap
- multispecies profile likelihood

## Makefile shortcuts

If the repository has the provided Makefile targets:

```bash
make smoke
make smoke-slow
```

## What smoke tests are for

Smoke tests are not a replacement for unit tests.

They check that the complete workflow stack runs:

```text
config -> CLI -> parser -> fitter -> export
```

They are intended to catch missing config files, broken CLI dispatch, renamed functions, changed config keys, broken export paths, and integration errors between modules.

## When to run smoke tests

Run fast smoke tests before pushing backend API changes, before handing code to the GUI collaborator, after changing CLI dispatch, after changing config formats, and after changing variable projection code.

Run slow smoke tests before tagging a release, before major demos, and before long GUI integration sessions.

## Expected output

A passing fast run ends with:

```text
Smoke test summary
==================
Passed: 8
Failed: 0
```

For slow smoke:

```text
Smoke test summary
==================
Passed: 12
Failed: 0
```

## If a smoke test fails

The runner prints workflow name, command, stdout, stderr, and failed workflow list.

Fix the first failure before continuing.

The most common causes are missing config file, wrong CLI branch, config key mismatch, fitter signature changed, or export function changed.
