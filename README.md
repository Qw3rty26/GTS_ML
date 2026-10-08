# GTS_ML

Galactic Tidal Stripping Simulations using [REBOUND](https://rebound.readthedocs.io/)

## Requirements

* Python
* [REBOUND](https://rebound.readthedocs.io/)

## Configuration

Simulation parameters are defined in `configuration.json`.

The configuration file controls the simulation setup, including:

* cluster generation
* number of stars
* integration settings
* timestep and softening
* output directory
* galactic potential
* orbital parameters

## Running

### Linux

```bash
./run.sh
```

### NixOS

```bash
./nixrun.sh
```

Both scripts use `configuration.json` for the simulation parameters.

Additional options can be passed from the command line:

```text
--config <path>
--verbose
--debug
```

For example:

```bash
./nixrun.sh --debug --config configuration.json
```

## Project structure

```text
GTS_ML/
├── py/                  # Simulation code
├── configuration.json   # Simulation configuration
├── run.sh               # Run using the local environment
└── nixrun.sh            # Run using Nix
```

## Output

Simulation results are written to the directory specified in `configuration.json`.

## Made by Qw3rty26

