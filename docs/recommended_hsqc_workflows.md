# Recommended HSQC / Global Observable Workflows

This document lists recommended command-line workflows for HSQC-style global observable fitting.

The intended data format is a wide CSV:

```text
time,peak_1,peak_2,peak_3,...
0,1000,850,1200,...
1,920,800,1100,...
2,830,740,980,...
```

Each peak column is treated as an observable reporting on one or more kinetic species.

## 1. Recommended default: variable projection fit

Use this when many peaks report on the same species.

Model:

```text
peak_i(t) = scale_i * A(t) + offset_i
```

Command:

```bash
python -m odefit.cli fit-global-observables   --config examples/configs/global_hsqc_variable_projection_config.json   --variable-projection
```

Use when each peak mostly tracks one species, you want fast fitting, and you have many peak-specific scale/offset terms.

## 2. Variable projection multistart

Use this for a more robust single-model fit.

```bash
python -m odefit.cli multistart-global-observables   --config examples/configs/global_hsqc_variable_projection_multistart_config.json   --variable-projection
```

## 3. Variable projection model comparison

Use this to compare candidate mechanisms.

```bash
python -m odefit.cli compare-global-observables   --config examples/configs/global_hsqc_variable_projection_model_comparison_config.json   --variable-projection
```

## 4. Variable projection multistart model comparison

This is the recommended robust mechanism-testing workflow.

```bash
python -m odefit.cli multistart-compare-global-observables   --config examples/configs/global_hsqc_variable_projection_multistart_model_comparison_config.json   --variable-projection
```

## 5. Bootstrap uncertainty

Use this after selecting a model.

```bash
python -m odefit.cli bootstrap-global-observables   --config examples/configs/global_hsqc_variable_projection_bootstrap_config.json   --variable-projection
```

Outputs include bootstrap parameter samples, bootstrap confidence intervals, bootstrap plots, and identifiability warnings.

## 6. Profile likelihood

Use this for stronger identifiability analysis.

```bash
python -m odefit.cli profile-likelihood-global-observables   --config examples/configs/global_hsqc_variable_projection_profile_likelihood_config.json   --variable-projection
```

## 7. Multispecies variable projection

Use this when each peak may report on a linear combination of species.

Model:

```text
peak_i(t) = c_iA*A(t) + c_iB*B(t) + ... + offset_i
```

```bash
python -m odefit.cli fit-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_config.json
```

## 8. Multispecies multistart

```bash
python -m odefit.cli fit-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_multistart_config.json
```

## 9. Multispecies model comparison

```bash
python -m odefit.cli compare-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_model_comparison_config.json
```

## 10. Multispecies multistart model comparison

This is the most robust multispecies mechanism-testing workflow.

```bash
python -m odefit.cli multistart-compare-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_multistart_model_comparison_config.json
```

## 11. Multispecies bootstrap

```bash
python -m odefit.cli bootstrap-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_bootstrap_config.json
```

## 12. Multispecies profile likelihood

```bash
python -m odefit.cli profile-likelihood-global-observables   --config examples/configs/global_hsqc_multispecies_variable_projection_profile_likelihood_config.json
```

## Practical recommended order for real data

1. Start with single-species variable projection.
2. Run single-species multistart.
3. Compare candidate mechanisms with multistart model comparison.
4. Run bootstrap on the selected model.
5. Run profile likelihood on key parameters.
6. If residuals suggest missing species contributions, try multispecies variable projection.
7. If multispecies improves fit meaningfully, repeat multistart model comparison and uncertainty analysis.

## Interpretation notes

BIC is usually the preferred default for model comparison because it penalizes additional parameters more strongly than AIC.

Bootstrap confidence intervals show uncertainty under residual resampling assumptions.

Profile likelihood is more informative for identifiability because it shows how much the fit worsens when a parameter is fixed away from its optimum.

Flat profile likelihood curves indicate weak identifiability.

Profile minima near grid boundaries suggest the parameter range should be widened or the parameter may be poorly constrained.
