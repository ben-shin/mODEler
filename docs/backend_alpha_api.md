# Backend alpha API

## Model parsing

Stable GUI-facing functions:
- build_model_spec(text)
- generate_ode_lines(model)

## Simulation

Stable GUI-facing functions:
- simulate_model(model, parameters, initial_conditions, timepoints, settings)

## Standard fitting

Stable GUI-facing functions:
- fit_model(...)
- export_fit_bundle(...)

## Global observable fitting

Stable GUI-facing functions:
- read_wide_observable_dataset(...)
- build_shared_species_observable_specs(...)
- fit_global_observable_model(...)
- fit_global_observable_multistart(...)

## CLI commands

Stable commands:
- generate-odes
- simulate
- fit
- multistart
- fit-global-observables
- multistart-global-observables
