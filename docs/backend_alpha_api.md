# Backend alpha API

This document describes the backend functions that are considered stable enough for GUI integration.

The GUI should collect user input, call these backend functions, display results, and export outputs. The GUI should not duplicate model parsing, simulation, fitting, plotting, or export logic.

This API is considered **alpha-stable**. Function names and core return types should not change casually after this point.

## Main backend principle

The GUI should act as a thin layer around the backend.

```text
GUI input widgets
    ↓
backend model/data/fitting functions
    ↓
backend result/export objects
    ↓
GUI display panels
```

Do not put scientific logic directly in the GUI.

---

# 1. Model parsing

## Build a model from reaction text

```python
from odefit.model.model_spec import build_model_spec

model = build_model_spec(
    text="""
A>B
B>C
"""
)
```

Returns:

```python
ModelSpec
```

Important fields:

```python
model.raw_text
model.reactions
model.species
model.parameters
model.name
model.metadata
model.warnings
```

GUI usage:

```text
User enters reaction text
GUI calls build_model_spec()
GUI displays species, parameters, warnings, and reactions
```

## Generate ODE text

```python
from odefit.model.ode_generator import generate_ode_lines

ode_lines = generate_ode_lines(model)
```

Returns:

```python
list[str]
```

GUI usage:

```text
Display generated ODEs in a read-only text panel
```

---

# 2. Reaction model syntax

Supported reaction syntax includes:

```text
A>B
A->B
A-B
A<->B
2A>A2
2A<->A2
A+B>C
A>B+C
label: A->B
```

Current limitations:

```text
Custom rate laws are not yet supported.
All reactions currently use mass-action kinetics.
```

Detailed syntax is documented in:

```text
docs/reaction_syntax.md
```

---

# 3. Dataset creation

## Standard direct-mapping dataset

```python
import pandas as pd

from odefit.data.dataset import Dataset

dataframe = pd.read_csv("data.csv")

dataset = Dataset(
    raw_dataframe=dataframe,
    time_column="time",
    signal_columns=["A", "B"],
)
```

Important fields:

```python
dataset.raw_dataframe
dataset.time_column
dataset.signal_columns
dataset.time_values
```

GUI usage:

```text
User opens CSV
GUI previews dataframe
User selects time column and signal columns
GUI builds Dataset
```

---

# 4. HSQC / wide observable datasets

For assigned HSQC peak intensity data in wide format:

```csv
time,A23_HN,G45_HN,L78_HN
0,1000,850,1200
1,920,810,1105
2,870,770,990
```

Use:

```python
from odefit.fitting.global_observables import (
    read_wide_observable_dataset_with_filtering,
)

dataset, filtering_result = read_wide_observable_dataset_with_filtering(
    file_path="hsqc_peaks.csv",
    time_column="time",
    signal_columns=None,
    exclude_columns=None,
    max_missing_fraction=0.25,
    min_initial_intensity=None,
    initial_points=1,
    min_dynamic_range=None,
    interpolate_missing=True,
)
```

Returns:

```python
Dataset
PeakFilteringResult
```

Important filtering result fields:

```python
filtering_result.kept_columns
filtering_result.removed_columns
filtering_result.removal_reasons
```

GUI usage:

```text
User opens HSQC peak table
GUI allows filtering settings
Backend filters peaks
GUI displays kept/removed peak table
```

To build a displayable filtering table:

```python
from odefit.data.peak_filtering import build_peak_filtering_table

peak_filtering_table = build_peak_filtering_table(filtering_result)
```

---

# 5. Simulation

## Simulate a model

```python
import numpy as np

from odefit.simulation.simulation_settings import SimulationSettings
from odefit.simulation.solver import simulate_model

settings = SimulationSettings(
    method="LSODA",
    rtol=1e-6,
    atol=1e-9,
    clip_negative_concentrations=False,
    warn_on_negative_values=True,
)

simulation_result = simulate_model(
    model=model,
    parameters={
        "k1f": 0.5,
    },
    initial_conditions={
        "A": 1.0,
        "B": 0.0,
    },
    timepoints=np.linspace(0.0, 10.0, 101),
    settings=settings,
)
```

Returns:

```python
SimulationResult
```

Important fields:

```python
simulation_result.timepoints
simulation_result.species
simulation_result.values
simulation_result.success
simulation_result.message
simulation_result.warnings
```

Get one species curve:

```python
a_values = simulation_result.get_species_values("A")
```

GUI usage:

```text
User enters parameter values and initial conditions
GUI calls simulate_model()
GUI plots species timecourses
GUI displays warnings
```

---

# 6. Parameter specs

Use `ParameterSpec` for fitted kinetic parameters.

```python
from odefit.fitting.parameter_spec import ParameterSpec

parameter_specs = [
    ParameterSpec(
        name="k1f",
        initial_guess=0.1,
        lower_bound=0.001,
        upper_bound=10.0,
    )
]
```

Important fields:

```python
name
initial_guess
lower_bound
upper_bound
fixed
fixed_value
tied_to
```

GUI usage:

```text
Display parameters in editable table
Allow user to edit initial guess and bounds
Allow fixed/tied parameters later
```

---

# 7. Initial condition specs

Use `InitialConditionSpec` for fitted or fixed initial species concentrations.

```python
from odefit.fitting.initial_condition_spec import InitialConditionSpec

initial_condition_specs = [
    InitialConditionSpec(
        species="A",
        initial_guess=1.0,
        lower_bound=0.0,
        upper_bound=2.0,
        fixed=True,
        fixed_value=1.0,
    ),
    InitialConditionSpec(
        species="B",
        initial_guess=0.0,
        lower_bound=0.0,
        upper_bound=2.0,
        fixed=True,
        fixed_value=0.0,
    ),
]
```

GUI usage:

```text
Display species initial conditions in editable table
Allow fixed or fitted initial conditions
```

---

# 8. Fit settings

```python
from odefit.fitting.fit_settings import FitSettings

settings = FitSettings(
    species_mapping={
        "A": "A",
        "B": "B",
    },
    use_normalized_data=False,
    method="trf",
    loss="linear",
    max_nfev=None,
    rtol=1e-6,
    atol=1e-9,
    signal_weights=None,
)
```

GUI usage:

```text
Collect mapping, optimizer, solver, and weighting options
Pass FitSettings into fitting functions
```

---

# 9. Standard direct fitting

For data columns that directly correspond to species:

```python
from odefit.fitting.optimizer import fit_model

fit_result = fit_model(
    model=model,
    dataset=dataset,
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    settings=settings,
)
```

Returns:

```python
FitResult
```

Important fields:

```python
fit_result.success
fit_result.message
fit_result.fitted_parameters
fit_result.initial_parameters
fit_result.fitted_initial_conditions
fit_result.initial_conditions
fit_result.fitted_observables
fit_result.statistics
fit_result.simulation_result
fit_result.residuals
fit_result.nfev
fit_result.cost
fit_result.status
fit_result.optimality
fit_result.active_mask
fit_result.njev
```

GUI usage:

```text
Run fit in worker thread
Display fitted parameters
Display fit statistics
Plot observed vs fitted
Plot residuals
Export bundle
```

---

# 10. Standard multistart fitting

```python
from odefit.fitting.multistart import fit_multistart

multistart_result = fit_multistart(
    model=model,
    dataset=dataset,
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    settings=settings,
    n_starts=20,
    random_seed=1,
    sort_by="aic",
    log_uniform=True,
)
```

Important fields:

```python
multistart_result.best_result
multistart_result.best_index
multistart_result.all_results
multistart_result.comparison_table
multistart_result.starting_parameter_sets
```

GUI usage:

```text
Run multiple fits
Show ranked comparison table
Display/export best fit
```

---

# 11. Parallel multistart fitting

```python
from odefit.fitting.parallel_multistart import fit_multistart_parallel

parallel_result = fit_multistart_parallel(
    model=model,
    dataset=dataset,
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    settings=settings,
    n_starts=40,
    n_workers=8,
    random_seed=1,
    sort_by="aic",
    log_uniform=True,
)
```

Important fields:

```python
parallel_result.successful_result
parallel_result.n_submitted
parallel_result.n_successful
parallel_result.n_failed
parallel_result.failures
```

GUI usage:

```text
Use for local CPU parallel fitting
Display progress/status
Display failures if any starts fail
```

---

# 12. Global observable fitting

Use this for many observed signals sharing one kinetic mechanism.

Example:

```text
peak_i(t) = scale_i * A(t) + offset_i
```

Build observable specs:

```python
from odefit.fitting.global_observables import (
    build_shared_species_observable_specs,
)

observable_specs = build_shared_species_observable_specs(
    signal_columns=dataset.signal_columns,
    species="A",
    fit_scale=True,
    fit_offset=True,
    scale_initial_guess=1.0,
    scale_lower_bound=0.0,
    scale_upper_bound=1000000.0,
    offset_initial_guess=0.0,
    offset_lower_bound=-1000000.0,
    offset_upper_bound=1000000.0,
)
```

Run global observable fit:

```python
from odefit.fitting.global_observables import fit_global_observable_model

output = fit_global_observable_model(
    model=model,
    dataset=dataset,
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    observed_species="A",
    settings=settings,
    observable_specs=observable_specs,
)

fit_result = output.fit_result
observable_specs = output.observable_specs
```

GUI usage:

```text
Use for HSQC peak fitting
Shared kinetic parameters
Peak-specific scale/offset
Display fitted observables table
```

---

# 13. Global observable multistart fitting

Use this for robust HSQC-style global fitting.

```python
from odefit.fitting.global_observable_multistart import (
    fit_global_observable_multistart,
)

global_multistart_result = fit_global_observable_multistart(
    model=model,
    dataset=dataset,
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    observable_specs=observable_specs,
    settings=settings,
    n_starts=20,
    n_workers=4,
    random_seed=1,
    sort_by="aic",
    log_uniform_parameters=True,
    randomize_observable_scales=True,
    randomize_observable_offsets=True,
    log_uniform_observable_scales=False,
)
```

Important fields:

```python
global_multistart_result.best_result
global_multistart_result.best_index
global_multistart_result.all_results
global_multistart_result.comparison_table
global_multistart_result.starting_parameter_sets
global_multistart_result.starting_observable_sets
global_multistart_result.failures
global_multistart_result.n_submitted
global_multistart_result.n_successful
global_multistart_result.n_failed
```

GUI usage:

```text
Use for robust multi-peak HSQC fitting
Display ranked starts
Display best fit
Display failed starts
Export comparison/start tables
```

---

# 14. Export fit bundle

Use this for all GUI exports.

```python
from odefit.export.bundle_export import export_fit_bundle

written_files = export_fit_bundle(
    fit_result=fit_result,
    model=model,
    dataset=dataset,
    output_dir="outputs/my_fit",
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    observable_specs=observable_specs,
    species_mapping=settings.species_mapping,
    include_plots=True,
    fit_settings=settings,
    command="gui-fit",
    config_path=None,
    extra_run_metadata={
        "source": "gui",
    },
)
```

Typical outputs:

```text
raw_data.csv
model_definition.txt
generated_odes.txt
fitted_parameters.csv
fitted_initial_conditions.csv
fitted_observables.csv
fit_statistics.csv
optimizer_diagnostics.csv
fit_diagnostics.csv
simulated_curves.csv
residuals.csv
model_metadata.json
dataset_metadata.json
parameter_specs.json
initial_condition_specs.json
observable_specs.json
fit_settings.json
fit_result_summary.json
run_metadata.json
```

GUI usage:

```text
Use this as the only export path for fit results
Do not manually duplicate export logic in GUI code
```

---

# 15. Project save/load

Use this for GUI project files.

## Create project state

```python
from odefit.project.project_state import create_project_state

project_state = create_project_state(
    project_name="my project",
    project_notes="notes",
    model_text="A>B",
    data_path="data.csv",
    time_column="time",
    signal_columns=["A", "B"],
    normalization_method="none",
    species_mapping={
        "A": "A",
        "B": "B",
    },
    parameter_specs=parameter_specs,
    initial_condition_specs=initial_condition_specs,
    observable_specs=observable_specs,
    fit_settings=settings,
    output_dir="outputs",
    last_fit_output_dir="outputs/latest",
    metadata={
        "experiment": "example",
    },
)
```

## Save project state

```python
from odefit.project.project_io import save_project_state

save_project_state(
    project_state=project_state,
    file_path="project.modeler.json",
)
```

## Load project state

```python
from odefit.project.project_io import load_project_state

project_state = load_project_state("project.modeler.json")
```

## Validate project state

```python
from odefit.project.project_io import (
    validate_project_state_for_fitting,
    validate_project_state_for_simulation,
)

validate_project_state_for_fitting(project_state)
validate_project_state_for_simulation(project_state)
```

GUI usage:

```text
File → Save Project
File → Open Project
Use validation before enabling fit/simulate buttons
```

---

# 16. Plotting

The GUI may use backend plotting helpers or render plots itself from result tables.

Useful backend plotting modules include:

```text
odefit.plotting.timecourse_plots
odefit.plotting.residual_plots
odefit.plotting.observed_vs_predicted
odefit.plotting.parameter_plots
odefit.plotting.diagnostics_plots
```

Recommended GUI approach:

```text
Use backend result objects and CSV tables as source of truth.
The GUI can render plots interactively, but exports should still use backend export functions.
```

---

# 17. CLI commands

Current stable CLI commands:

```text
generate-odes
simulate
fit
multistart
fit-global-observables
multistart-global-observables
```

Examples:

```bash
python -m odefit.cli generate-odes \
  --model examples/configs/model_first_order.txt
```

```bash
python -m odefit.cli simulate \
  --config examples/configs/simulation_config.json
```

```bash
python -m odefit.cli fit \
  --config examples/configs/fit_config.json
```

```bash
python -m odefit.cli multistart \
  --config examples/configs/multistart_config.json
```

```bash
python -m odefit.cli fit-global-observables \
  --config examples/configs/global_hsqc_fit_config.json
```

```bash
python -m odefit.cli multistart-global-observables \
  --config examples/configs/global_hsqc_multistart_config.json
```

---

# 18. Recommended GUI alpha workflow

The first GUI alpha should implement this path:

```text
1. Open CSV
2. Preview data
3. Choose time column
4. Choose signal columns
5. Enter reaction model text
6. Parse model
7. Display species, parameters, warnings, and generated ODEs
8. Edit parameter guesses/bounds
9. Edit initial conditions
10. Run simulation
11. Run fit
12. Display fitted parameters and statistics
13. Plot observed vs fitted
14. Plot residuals
15. Export fit bundle
16. Save project
17. Load project
```

Do not implement every advanced feature in the first GUI alpha.

Recommended first GUI tabs:

```text
Project
Data
Model
Parameters
Simulation
Fitting
Results
Export
```

---

# 19. Things not yet alpha-stable

The following should not be treated as stable GUI-facing features yet:

```text
Custom rate laws
Michaelis-Menten syntax
Hill syntax
Linear-combination observables from CLI
Long-format HSQC input
Multi-dataset global fitting
Bootstrap confidence intervals
Profile likelihood
GPU/JAX backend
Julia backend
SLURM/cluster submission
```

These should be added after the backend alpha and GUI alpha are working.

---

# 20. Backend development rules from this point

1. Do not change public function names without updating this document.
2. Do not change dataclass field names without migration or compatibility handling.
3. New GUI-facing functionality should have tests.
4. New CLI functionality should have config examples.
5. Export bundle format should remain backward-compatible where possible.
6. Project file schema changes must increment `PROJECT_STATE_SCHEMA_VERSION`.
7. GUI should call backend functions rather than duplicate backend logic.

