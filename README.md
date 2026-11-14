# GTS_ML
### Galactic Tidal Stripping Simulations with Machine Learning

GTS_ML is a Python project for simulating the tidal stripping of stellar clusters in a galactic potential and using the resulting simulations to train machine-learning models.

The project combines numerical N-body simulations, dataset generation, and classification to study how the initial orbital parameters of a stellar cluster influence its survival.

## Physical model

Both the stellar cluster and the galaxy are modelled as Plummer spheres. The simulations are performed using [REBOUND](https://rebound.readthedocs.io/), an N-body simulation package.

Each simulation starts from a generated stellar cluster placed on an orbit around the galactic potential. The main orbital parameters are:

- `orbit_semi_major_axis`: orbital semi-major axis.
- `orbit_eccentricity`: orbital eccentricity.

During the simulation, the evolution of the cluster is tracked through several physical quantities, including its bound mass fraction and half-mass radius.

## Example simulations

The following examples show two different outcomes of the tidal stripping process. The simulations are visualized using [OVITO](https://www.ovito.org/).

| Surviving cluster | Dissolved cluster |
|:---:|:---:|
| ![Surviving cluster](Docs/images/cluster_alive.png) | ![Dissolved cluster](Docs/images/cluster_dead.png) |
| `a = 300.0`, `e = 0.5` | `a = 140.0`, `e = 0.8` |
| Bound mass fraction: `0.93` | Bound mass fraction: `0.065` |
| Relative half-mass radius: `1.002` | Relative half-mass radius: `0.246` |

## Requirements

The project requires Python and the packages listed in `requirements.txt`:

- NumPy
- SciPy
- Numba
- REBOUND

## Installation and execution

Two scripts are provided to run the project:

**Linux**

```bash
./run.sh
```

**NixOS**

```bash
./nixrun.sh
```

Both scripts use `configuration.json` by default. A different configuration file can be specified with `--config`, while `--debug` enables debug-level logging.

```bash
./run.sh --config path/to/configuration.json
./run.sh --debug
```

The execution consists of three stages:

1. **Cluster generation** — generates initial stellar clusters using the Plummer model and saves their snapshots.
2. **Galactic tidal stripping** — simulates each cluster in the galactic potential, tracking its evolution over time.
3. **Dataset generation** — extracts selected initial and final diagnostics from the simulations and saves them in a CSV file.

The output directory and the locations of the generated files are printed during execution.

## Configuration

Simulation parameters are defined in `configuration.json`.

### Simulation configuration

| Parameter | Description |
|---|---|
| `dt` | Integration timestep. |
| `t` | Initial simulation time. |
| `G` | Gravitational constant. |
| `softening` | Gravitational softening length. |
| `integrator` | REBOUND integration method. |

### Cluster generation

| Parameter | Description |
|---|---|
| `master_seed` | Seed used to generate the cluster realizations. |
| `number_of_clusters` | Number of initial cluster realizations. |
| `number_of_particles` | Number of stars in each cluster. |
| `cluster_radius` | Initial Plummer scale radius. |
| `snapshot_interval` | Time interval between saved snapshots. |

### Galactic tidal stripping

| Parameter | Description |
|---|---|
| `galaxy_mass` | Mass of the galactic Plummer model. |
| `galaxy_radius` | Scale radius of the galactic Plummer model. |
| `orbit_semi_major_axis` | List of orbital semi-major axes to simulate. |
| `orbit_eccentricity` | List of orbital eccentricities to simulate. |
| `number_of_orbits` | Maximum number of completed orbits. |
| `end_time` | Maximum simulation time. |
| `snapshot_interval` | Time interval between saved snapshots. |
| `simulate_isolated_cluster` | Whether to run a corresponding isolated-cluster simulation. |

At least one of `number_of_orbits` and `end_time` must be enabled. Setting either parameter to `-1` disables that stopping condition. If both are enabled, the simulation stops when the first condition is reached.

### Other parameters

| Parameter | Description |
|---|---|
| `cores_used` | Number of CPU cores used for parallel simulations. |
| `output_directory` | Directory where simulation results are stored. |

## Simulation output

Each execution creates a `run_###/` directory inside the configured output directory. Its contents are organized as follows:

```text
run_###/
├── GEN/
│   └── ...                  # Initial cluster snapshots
├── GTS/
│   └── ...                  # Galactic and isolated simulations
└── machine_learning_dataset.csv
```

The `GEN/` directory contains the generated initial cluster realizations. The `GTS/` directory contains the galactic tidal stripping simulations and, when enabled, their isolated counterparts.

Simulation snapshots are saved in JSON format. Particle positions and velocities can also be stored in XYZV format for visualization with OVITO. Diagnostic data are saved as comma-separated CSV files.

The diagnostics include:

- Bound mass fraction.
- Half-mass radius and its relative value.
- Kinetic and potential energies.
- Virial ratio.
- Relative energy error.
- Angular momentum.
- Number of completed orbits.

When isolated-cluster simulations are enabled, their diagnostic CSV files are saved separately and excluded from the machine-learning dataset.

## Dataset and machine learning

The dataset is generated from the initial and final diagnostics of the galactic simulations. Each row represents one simulation and contains its initial orbital parameters and selected physical quantities.

The dataset used in the thesis contains **17,100 simulations**, corresponding to **1,140 orbital configurations** and **15 initial cluster realizations** per configuration.

The classification task is to predict whether a stellar cluster survives or dissolves. The input features are:

- Orbital semi-major axis.
- Orbital eccentricity.

A cluster is classified as surviving when both conditions are satisfied:

- Bound mass fraction greater than `0.1`.
- Relative half-mass radius greater than `0.2`.

Two classifiers were evaluated:

- **Support Vector Classifier (SVC)** with a degree-2 polynomial kernel.
- **Multi-Layer Perceptron (MLP)** with two hidden layers containing 16 and 8 neurons.

The dataset was split into training and test sets using an 80/20 stratified split.

| Model | Weighted F1-score |
|---|---:|
| SVC | 0.9644 |
| MLP | 0.9642 |

The generated dataset is saved as `machine_learning_dataset.csv` in the root of each `run_###/` directory.

## Thesis and presentation

The bachelor's thesis and its accompanying presentation are available in the `Docs/` directory:

```text
Docs/
├── asaro_valerio_laurea_26.pdf
├── asaro_valerio_laurea_26_eng.pdf
└── presentation/
    └── ...
```

- `asaro_valerio_laurea_26.pdf` — bachelor's thesis in Italian.
- `asaro_valerio_laurea_26_eng.pdf` — English translation of the thesis.
- `presentation/` — presentation slides, available in Italian only.

## Documentation and analysis

Additional material is organized in the following directories:

```text
Docs/
├── images/              # Images used in the documentation
├── plots/               # Plots generated during the analysis
└── jupyter-notebooks/   # Notebooks for data analysis and model evaluation
```

## Project structure

```text
GTS_ML/
├── py/                         # Python source code
├── Docs/
│   ├── images/
│   ├── plots/
│   ├── jupyter-notebooks/
│   ├── presentation/
│   ├── asaro_valerio_laurea_26.pdf
│   └── asaro_valerio_laurea_26_eng.pdf
├── configuration.json
├── requirements.txt
├── run.sh
├── nixrun.sh
└── README.md
```

## Release

**Final public release**
