# Global observable fitting

This is designed for experiments where many signals report on the same kinetic mechanism.

A typical example is a 2D HSQC NMR timecourse of a protein with multiple peaks.

Each peak has its own profile, but all peaks are assumed to share the same underlying kinetic model.

## Example use case

Suppose an experiment gives assigned peaks measured over time:

```csv
time,A23,G45,L78,V90
0,2.1000,1.7000,0.8500,3.2000
1,1.4406,1.2055,0.5862,2.2109
2,0.9987,0.8740,0.4095,1.5481

The model may be A>B, where A is the NMR visible monomer and B is a larger/invisible state.

The global observable model is:

peak_i(t) = scale_i * A(t) + offset_i

Meaning k1,A(t),B(t), and initial conditions are shared globally.

# Command line usage

Run
```bash
python -m odefit.cli fit-global-observables \
  --config examples/configs/global_hsqc_fit_config.json

Example config
```json
{
  "model": "examples/configs/model_first_order.txt",
  "data": "examples/configs/example_hsqc_peaks.csv",
  "time_column": "time",
  "observed_species": "A",
  "parameters": {
    "k1f": {
      "initial_guess": 0.1,
      "lower_bound": 0.000001,
      "upper_bound": 10.0
    }
  },
  "initial_conditions": {
    "A": {
      "value": 1.0,
      "mode": "fixed",
      "lower_bound": 0.0,
      "upper_bound": 2.0
    },
    "B": {
      "value": 0.0,
      "mode": "fixed",
      "lower_bound": 0.0,
      "upper_bound": 2.0
    }
  },
  "fit_scale": true,
  "fit_offset": true,
  "scale_initial_guess": 1.0,
  "scale_lower_bound": 0.0,
  "scale_upper_bound": 1000000.0,
  "offset_initial_guess": 0.0,
  "offset_lower_bound": -1000000.0,
  "offset_upper_bound": 1000000.0,
  "method": "trf",
  "loss": "linear",
  "rtol": 1e-8,
  "atol": 1e-10,
  "max_nfev": 5000,
  "output_dir": "examples/configs/outputs/global_hsqc_fit",
  "no_plots": true
}


