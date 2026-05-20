# Engine Benchmark Summary

Baseline engine: `reference`

## Available benchmark rows

| benchmark | engine_name | n_timepoints | n_observables | n_repeats | median_seconds | mean_seconds | speedup_vs_reference_median | mean_nfev | median_k1f | median_rss | rss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| numpy_loop | numpy_loop | 50 | 1000 | 20 | 0.0122806 | 0.0129001 |  |  |  |  |  |
| numpy_vectorized | numpy_vectorized | 50 | 1000 | 20 | 7.1831e-05 | 8.0414e-05 |  |  |  |  |  |
| project_single_species_batch | jax_projection | 50 | 1000 | 50 | 0.000351783 | 0.000390011 | 0.215397 |  |  |  | 0.19122 |
| project_single_species_batch | numba_projection | 50 | 1000 | 50 | 5.7324e-05 | 6.56397e-05 | 1.32184 |  |  |  | 0.19122 |
| project_single_species_batch | reference | 50 | 1000 | 50 | 7.5773e-05 | 8.77213e-05 | 1 |  |  |  | 0.19122 |
| single_species_projection | jax_projection | 100 |  | 100 | 0.000195942 | 0.000212778 | 0.0646389 |  |  |  |  |
| single_species_projection | numba_projection | 100 |  | 100 | 4.8025e-06 | 5.42399e-06 | 2.63727 |  |  |  |  |
| single_species_projection | reference | 100 |  | 100 | 1.26655e-05 | 1.44926e-05 | 1 |  |  |  |  |
| single_species_variable_projection_fit | jax_projection |  |  | 5 | 0.159738 | 0.167271 | 0.83156 | 7 | 0.39994 | 0.0113674 |  |
| single_species_variable_projection_fit | numba_projection |  |  | 5 | 0.125 | 0.135009 | 1.06265 | 5 | 0.39994 | 0.0113674 |  |
| single_species_variable_projection_fit | reference |  |  | 5 | 0.132831 | 0.132419 | 1 | 5 | 0.39994 | 0.0113674 |  |

## Unavailable engines

No unavailable engines were reported.

## Notes

- Speedup values are computed as reference median time divided by engine median time.
- Values greater than 1 mean faster than reference.
- Workflow-level benchmarks include API/config overhead, ODE solving, optimizer calls, and projection.
- Kernel-level benchmarks isolate smaller pieces and may exaggerate apparent speedups.
