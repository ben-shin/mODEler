# Suggested README Section: Backend and GUI API

## Backend API for GUI Integration

mODEler provides a GUI-facing backend API under:

```python
odefit.api.backend
odefit.api.serialization
```

The GUI should use this API rather than importing low-level fitting modules directly.

Main functions:

```python
parse_model_text(...)
simulate_from_text(...)
fit_global_observables_from_config(...)
compare_global_observables_from_config(...)
bootstrap_global_observables_from_config(...)
profile_likelihood_global_observables_from_config(...)
backend_output_payload(...)
```

See:

```text
docs/backend_api_v0_2.md
docs/gui_integration_guide.md
docs/recommended_hsqc_workflows.md
docs/hsqc_config_reference.md
docs/backend_smoke_tests.md
```

Recommended smoke tests:

```bash
python scripts/smoke_test_backend_workflows.py --stop-on-failure
python scripts/smoke_test_backend_workflows.py --include-slow --stop-on-failure
```
