# GUI Integration Guide

This guide explains how the mODEler GUI should interact with the backend.

The GUI should behave as a thin client over the backend. It should not reimplement model parsing, ODE generation, simulation, fitting, variable projection, multistart fitting, model comparison, bootstrap, profile likelihood, or export logic.

The GUI should gather user inputs, build backend config dictionaries, call backend API functions, and display serialized payloads.

## Recommended GUI tabs

A practical GUI alpha could be organized as:

1. Data
2. Model
3. Parameters
4. Initial Conditions
5. Simulation
6. Fit
7. Model Comparison
8. Uncertainty
9. Export

## 1. Data tab

### User actions

- Open CSV
- Preview table
- Select time column
- Select signal columns
- Exclude columns
- Configure peak filtering

### Backend config keys

```json
{
  "data": "path/to/data.csv",
  "time_column": "time",
  "signal_columns": null,
  "exclude_columns": null,
  "max_missing_fraction": 0.0,
  "min_initial_intensity": null,
  "initial_points": 1,
  "min_dynamic_range": null,
  "interpolate_missing": true
}
```

### GUI display

Show raw table preview, detected columns, kept signal columns, removed signal columns, and filtering warnings.

## 2. Model tab

### User actions

- Enter model text
- Parse model
- View detected species
- View detected parameters
- View generated ODEs

### Backend call

```python
from odefit.api.backend import parse_model_text

parsed = parse_model_text(model_text)
```

### GUI display

Show species list, parameter list, reaction list, and generated ODEs.

## 3. Parameters tab

The GUI should auto-populate parameter rows from `parse_model_text`.

Each parameter row should include name, initial guess, lower bound, and upper bound.

Example config:

```json
{
  "parameters": {
    "k1f": {
      "initial_guess": 0.1,
      "lower_bound": 0.000001,
      "upper_bound": 10.0
    }
  }
}
```

## 4. Initial Conditions tab

The GUI should auto-populate species rows from `parse_model_text`.

Each species row should include species, value, mode, lower bound, and upper bound.

Example config:

```json
{
  "initial_conditions": {
    "A": {
      "value": 1.0,
      "mode": "fixed"
    },
    "B": {
      "value": 0.0,
      "mode": "fixed"
    }
  }
}
```

## 5. Simulation tab

### Backend call

```python
from odefit.api.backend import simulate_from_text

simulation = simulate_from_text(
    model_text=model_text,
    parameters={"k1f": 0.4},
    initial_conditions={"A": 1.0, "B": 0.0},
    timepoints=[0, 1, 2, 3, 4, 5],
)
```

### GUI display

Plot species vs time and optionally show a table preview.

## 6. Fit tab

### Recommended default for HSQC

Use variable projection.

For single-species HSQC-style decay:

```json
{
  "use_variable_projection": true,
  "observed_species": "A",
  "fit_scale": true,
  "fit_offset": true
}
```

For multispecies observables:

```json
{
  "use_multispecies_variable_projection": true,
  "observed_species": ["A", "B"],
  "fit_offset": true
}
```

### Backend call

```python
from odefit.api.backend import fit_global_observables_from_config
from odefit.api.serialization import backend_output_payload

output = fit_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="fit")
```

### GUI display

Show fit success, message, fitted kinetic parameters, statistics, predicted vs observed curves, residuals, and observable coefficients.

## 7. Model Comparison tab

### User actions

- Add candidate model
- Name each model
- Edit parameter guesses/bounds per model
- Edit initial conditions per model
- Choose ranking criterion

Default ranking:

```json
"sort_by": "bic"
```

Allowed ranking fields:

```text
rss
rmse
aic
bic
```

### Backend call

```python
from odefit.api.backend import compare_global_observables_from_config
from odefit.api.serialization import backend_output_payload

output = compare_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="model_comparison")
```

### GUI display

Show ranked comparison table, best model, best model fit statistics, and failed models if any.

## 8. Uncertainty tab

Uncertainty should support bootstrap, profile likelihood, and identifiability warnings.

### Bootstrap backend call

```python
from odefit.api.backend import bootstrap_global_observables_from_config
from odefit.api.serialization import backend_output_payload

output = bootstrap_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="bootstrap")
```

### Profile likelihood backend call

```python
from odefit.api.backend import profile_likelihood_global_observables_from_config
from odefit.api.serialization import backend_output_payload

output = profile_likelihood_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="profile_likelihood")
```

### GUI display for bootstrap

Show bootstrap parameter summary table, bootstrap parameter histograms, bootstrap confidence intervals, number of successful replicates, and number of failed replicates.

### GUI display for profile likelihood

Show profile likelihood curve per parameter, profile table, and warnings about flat profiles or boundary minima.

## 9. Export tab

The GUI should offer save project config, export fit bundle, export tables, export plots, and export serialized backend payload.

For alpha GUI, saving a JSON config is enough.

## Minimal GUI alpha workflow

A useful first GUI alpha should support:

1. Open CSV
2. Preview data
3. Select time column
4. Select signal columns
5. Enter model text
6. Parse model
7. Edit parameter guesses/bounds
8. Edit initial conditions
9. Run variable projection fit
10. Plot observed vs fitted
11. Plot residuals
12. Export config and results

Everything else can come after.

## Error handling

The GUI should catch backend exceptions and show error type, error message, relevant config section, and full traceback in an optional developer panel. Do not hide errors silently.

## GUI should not depend on internals

Avoid importing from:

```python
odefit.fitting.*
odefit.model.*
odefit.simulation.*
```

Prefer:

```python
odefit.api.backend
odefit.api.serialization
```

This keeps GUI code stable while backend internals evolve.
