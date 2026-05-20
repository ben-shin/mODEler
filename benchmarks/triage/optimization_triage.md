# Engine Optimization Triage

Input summary: `benchmarks/summary/engine_benchmark_summary.csv`

## Findings

### 1. Available benchmarked engines

- Category: `engine_coverage`
- Severity: `info`
- Evidence: jax_projection, numba_projection, numpy_loop, numpy_vectorized, reference
- Recommendation: Keep reference as the baseline. Treat optional engines as experimental until workflow-level benchmarks show stable gains.

### 2. Projection kernel speedup measured

- Category: `projection_kernel`
- Severity: `info`
- Evidence: Best non-reference projection-kernel speedup: 2.64x.
- Recommendation: Kernel speedup is useful, but do not overinterpret it. Projection may be only a small fraction of full fitting time.

### 3. Production batch projection speedup measured

- Category: `batched_projection`
- Severity: `medium`
- Evidence: Best non-reference batch-method speedup: 1.32x.
- Recommendation: Batched projection appears worth productionizing further. Next targets: reduce residual DataFrame construction and avoid per-evaluation pandas overhead in variable projection.

### 4. Workflow speedup is marginal

- Category: `workflow`
- Severity: `medium`
- Evidence: Best non-reference workflow speedup: 1.06x.
- Recommendation: Projection acceleration is not yet dominating end-to-end time. Next target should be profiling residual assembly, ODE solving, and optimizer callback overhead.

### 5. Kernel speedup does not translate to workflow speedup

- Category: `bottleneck_inference`
- Severity: `high`
- Evidence: Kernel speedup is 2.64x, but workflow speedup is only 1.06x.
- Recommendation: The bottleneck is likely outside the scalar projection kernel. Prioritize timing residual assembly, ODE simulation calls, DataFrame construction, and scipy least_squares callback overhead.

## Suggested next optimization order

1. Confirm full workflow benchmark coverage.
2. If kernel speedup does not translate to workflow speedup, profile callback overhead and residual assembly.
3. If batch projection speedup is strong, wire larger batched operations into the workflow.
4. Only move toward ODE/JAX/GPU solver work after projection and residual assembly are no longer obvious bottlenecks.
