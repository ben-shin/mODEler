# mODEller
mODEller is a GUI-based software package for:

- constructing kinetic reaction models
- automatically generating ODE systems
- fitting ODE models to experimental data
- visualizing simulation and fitting results
- exporting plots, statistics, and fitted parameters

## Features

- CSV data importing
- Automatic reaction parsing
- Automatic ODE generation
- Parameter fitting
- Raw and normalized data handling
- Multiple solver backends
- Plotting and support

## Example reaction syntax

```text
2P1-P2
P1+P2-P3
P2+P3-P5
P5>P_proto
```

## Installation

```bash
git clone git@github.com:ben-shin/mODEller.git
cd mODEller

conda env create -f environment.yml
conda activate modeller
```

## Running

```bash
python -m odefit.app
```

## Development

```bash
pytest tests/
```

## Shoutout
Shoutout goes to Sebastian for breaking his arm cycling
I've always been meaning to develop something like this
as ODEs are typically hard to approach for many initially
and needed an extra pair of hands and another brain to
make this happen. In this case an extra hand and another
brain. But thanks again for agreeing to it in principle
and donating your time big dog
