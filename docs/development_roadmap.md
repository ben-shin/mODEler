# mODEler Development Roadmap

This roadmap tracks the remaining work after the backend alpha and HSQC/global observable fitting expansion.

## Completed major milestones

```text
[x] Reaction parser
[x] ODE generation
[x] Simulation
[x] Basic fitting
[x] Multistart fitting
[x] Global observable fitting
[x] Global observable multistart
[x] Model comparison
[x] Multistart model comparison
[x] Project save/load basics
[x] Backend alpha API documentation
[x] Progress reporting and ETA
[x] Backend capability detection
[x] Benchmark utilities
[x] Variable projection fitting
[x] Variable projection multistart
[x] Variable projection model comparison
[x] Variable projection multistart model comparison
[x] Bootstrap uncertainty
[x] Profile likelihood
[x] Identifiability diagnostics
[x] Multispecies variable projection
[x] Multispecies variable projection multistart
[x] Multispecies variable projection model comparison
[x] Multispecies variable projection multistart model comparison
[x] Multispecies bootstrap
[x] Multispecies profile likelihood
[x] GUI-facing backend API v0.2
[x] GUI-friendly serialization
[x] Backend smoke tests
```

## Immediate next phase: GUI alpha support

```text
[ ] Finalize backend API v0.2 docs
[ ] Add API example notebooks or scripts
[ ] Add GUI contract examples
[ ] Add saved project schema v0.2
[ ] Add project save/load coverage for new workflows
[ ] Hand stable backend contract to GUI collaborator
```

## GUI alpha target

A useful GUI alpha should support:

```text
[ ] Open CSV
[ ] Preview data
[ ] Select time column
[ ] Select signal columns
[ ] Enter model text
[ ] Parse model
[ ] Show species
[ ] Show parameters
[ ] Show generated ODEs
[ ] Edit parameter guesses/bounds
[ ] Edit initial conditions
[ ] Run simulation
[ ] Run variable projection fit
[ ] Plot observed vs fitted
[ ] Plot residuals
[ ] Export config
[ ] Export results
```

Secondary GUI features:

```text
[ ] Multistart fitting
[ ] Model comparison
[ ] Bootstrap uncertainty
[ ] Profile likelihood
[ ] Identifiability warnings
[ ] Multispecies variable projection
```

## Scientific robustness

```text
[ ] Add real HSQC tutorial dataset
[ ] Add synthetic truth datasets
[ ] Add amyloid aggregation examples
[ ] Add benchmark accuracy tests
[ ] Add reproducibility metadata
[ ] Add environment capture
[ ] Add version metadata in all exports
```

## Documentation

```text
[ ] Full user tutorial
[ ] CLI reference
[ ] GUI integration docs
[ ] Config reference
[ ] Scientific examples
[ ] Troubleshooting guide
```

## Performance and acceleration

Do not prioritize GPU/JAX/Julia/Cython until backend workflows and GUI-facing API are stable.

Near-term performance priorities:

```text
[ ] Parallel model comparison
[ ] Parallel profile likelihood
[ ] Better progress reporting for all long workflows
[ ] Benchmark large HSQC datasets
```

Later acceleration priorities:

```text
[ ] Backend abstraction layer
[ ] JAX prototype backend
[ ] Batched multistart
[ ] Batched bootstrap
[ ] GPU acceleration for batched workflows
[ ] Cython only for proven hot kernels
[ ] Julia bridge only if stiff/large systems demand it
```

## Advanced modelling

```text
[ ] Custom rate laws
[ ] Symbolic validation
[ ] User-defined observables
[ ] Bounded multispecies projection
[ ] NNLS projection backend
[ ] Multi-dataset global fitting
[ ] Shared/global parameter constraints
[ ] Bayesian/MCMC fitting
```

## Release engineering

```text
[ ] Semantic versioning
[ ] PyPI packaging
[ ] Release notes
[ ] Test matrix
[ ] Documentation site
[ ] Example gallery
```

## Recommended order from here

1. Finish and commit documentation bundle.
2. Run unit tests.
3. Run fast smoke tests.
4. Run slow smoke tests.
5. Give backend API v0.2 docs to GUI collaborator.
6. Begin GUI alpha.
7. Add project schema v0.2 once GUI needs save/load.
8. Only then revisit JAX/GPU acceleration.
