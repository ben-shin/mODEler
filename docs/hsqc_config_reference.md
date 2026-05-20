# HSQC / Global Observable Config Reference

This document describes the main JSON config keys used by mODEler HSQC/global observable workflows.

## Common fields

```json
{
  "data": "examples/configs/example_hsqc_peaks.csv",
  "time_column": "time",
  "signal_columns": null,
  "exclude_columns": null,
  "output_dir": "examples/configs/outputs/my_run"
}
```

### `data`

Path to the input CSV file.

### `time_column`

Name of the time column.

### `signal_columns`

List of observable columns to include. If `null`, all non-time columns are used unless excluded.

### `exclude_columns`

List of columns to exclude.

### `output_dir`

Directory where outputs are written.

## Model fields

Single-model workflows use either:

```json
{
  "model": "examples/configs/model_first_order.txt"
}
```

or:

```json
{
  "model_text": "A -> B"
}
```

Model comparison workflows use:

```json
{
  "model_texts": {
    "single_step": "A -> B",
    "two_step": "A -> B
B -> C"
  }
}
```

or:

```json
{
  "model_files": {
    "single_step": "models/single_step.txt",
    "two_step": "models/two_step.txt"
  }
}
```

## Parameter fields

Single-model workflows:

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

Model comparison workflows:

```json
{
  "parameters_by_model": {
    "single_step": {
      "k1f": {
        "initial_guess": 0.1,
        "lower_bound": 0.000001,
        "upper_bound": 10.0
      }
    },
    "two_step": {
      "k1f": {
        "initial_guess": 0.1,
        "lower_bound": 0.000001,
        "upper_bound": 10.0
      },
      "k2f": {
        "initial_guess": 0.05,
        "lower_bound": 0.000001,
        "upper_bound": 10.0
      }
    }
  }
}
```

Fallback defaults:

```json
{
  "default_parameter_guess": 0.1,
  "default_parameter_lower": 0.0,
  "default_parameter_upper": 100.0
}
```

## Initial condition fields

Single-model workflows:

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

Model comparison workflows:

```json
{
  "initial_conditions_by_model": {
    "single_step": {
      "A": {
        "value": 1.0,
        "mode": "fixed"
      },
      "B": {
        "value": 0.0,
        "mode": "fixed"
      }
    },
    "two_step": {
      "A": {
        "value": 1.0,
        "mode": "fixed"
      },
      "B": {
        "value": 0.0,
        "mode": "fixed"
      },
      "C": {
        "value": 0.0,
        "mode": "fixed"
      }
    }
  }
}
```

Initial condition modes:

```text
fixed
fit
```

If `mode` is `fixed`, the value is held constant. If `mode` is `fit`, the value can be optimized.

## Single-species variable projection

```json
{
  "use_variable_projection": true,
  "observed_species": "A",
  "fit_scale": true,
  "fit_offset": true
}
```

Model:

```text
peak_i(t) = scale_i * A(t) + offset_i
```

## Multispecies variable projection

```json
{
  "use_multispecies_variable_projection": true,
  "observed_species": ["A", "B"],
  "fit_offset": true
}
```

Model:

```text
peak_i(t) = c_iA*A(t) + c_iB*B(t) + offset_i
```

For model comparison, use:

```json
{
  "observed_species_by_model": {
    "single_step": ["A", "B"],
    "two_step": ["A", "B"]
  }
}
```

## Optimizer settings

```json
{
  "method": "trf",
  "loss": "linear",
  "max_nfev": null,
  "rtol": 0.000001,
  "atol": 0.000000001
}
```

## Variable projection backend settings

```json
{
  "variable_projection_backend": "numpy",
  "variable_projection_method": "LSODA"
}
```

Currently recommended:

```json
{
  "variable_projection_backend": "numpy",
  "variable_projection_method": "LSODA"
}
```

## Peak filtering fields

```json
{
  "max_missing_fraction": 0.0,
  "min_initial_intensity": null,
  "initial_points": 1,
  "min_dynamic_range": null,
  "interpolate_missing": true
}
```

## Multistart fields

```json
{
  "use_multistart": true,
  "n_starts": 10,
  "random_seed": 123,
  "sort_by": "bic",
  "multistart_sort_by": "bic",
  "log_uniform": true
}
```

Allowed ranking fields:

```text
rss
rmse
aic
bic
```

## Bootstrap fields

```json
{
  "n_bootstrap": 100,
  "n_workers": 4,
  "random_seed": 123,
  "confidence_level": 0.95,
  "show_progress": true,
  "no_plots": false
}
```

## Profile likelihood fields

```json
{
  "profile_parameters": ["k1f"],
  "profile_n_points": 15,
  "profile_span_factor": 10.0,
  "profile_log_space": true,
  "no_plots": false
}
```

## Recommended config defaults

For most HSQC workflows:

```json
{
  "sort_by": "bic",
  "method": "trf",
  "loss": "linear",
  "rtol": 0.000001,
  "atol": 0.000000001,
  "variable_projection_backend": "numpy",
  "variable_projection_method": "LSODA",
  "fit_offset": true,
  "interpolate_missing": true
}
```
