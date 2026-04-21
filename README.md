# Information Routing in Multi-Frequency Oscillatory Networks

This repository contains C-based Hodgkin-Huxley network simulations and Python
analysis pipelines for studying oscillatory regimes, MFOPs, transfer entropy,
and spike transmission across single-population and connected-population
networks.

## Repository Tree

```text
.
|-- README.md
|-- pyproject.toml
|-- include/
|   |-- Makefile
|   |-- model2.{c,h}
|   |-- neuralnet.{c,h}
|   |-- measurement2.{c,h}
|   |-- storage.{c,h}
|   |-- mpifor.{c,h}
|   |-- rng.{c,h}
|   |-- mt64.{c,h}
|   |-- ntk.{c,h}
|   `-- utils.{c,h}
|-- lib/
|   `-- libhhnet.a
|-- simulations/
|   |-- compile.sh
|   |-- descript.md
|   |-- main_single.c
|   |-- main_single_2d.c
|   |-- main_twopop.c
|   |-- main_twopop_4d.c
|   |-- main_monopop_4d.c
|   |-- main_regime_samples.c
|   |-- main_transmission.c
|   |-- export_regime_params.py
|   `-- export_transmission_params.py
|-- analysis/
|   |-- export_singlepop_summary.py
|   |-- extract_burstprobs_singlepop.py
|   |-- export_twopop_summary.py
|   |-- run_kmeans_clustering.py
|   |-- clustering_oscillation_features.ipynb
|   |-- compute_coburst_probs.py
|   |-- extract_burstprobs_twopop.py
|   |-- determine_frequency_range.ipynb
|   |-- identify_mfop.py
|   |-- identify_mfop_transmission.py
|   |-- convert_mua.py
|   |-- compute_te.py
|   |-- compute_kappa.py
|   `-- compute_recv_response.py
|-- src/pytools/
|   |-- hhtools.py
|   |-- hhsignal.py
|   |-- hhsummary.py
|   |-- hhclustering.py
|   |-- hhfilter.py
|   |-- hhinfo.py
|   |-- tetools.py
|   |-- oscdetector.py
|   |-- power_utils.py
|   |-- burst_tools.py
|   |-- utils*.py
|   `-- visu.py
|-- figures/
|   |-- main_figure1.py
|   |-- main_figure2.py
|   |-- main_figure3.py
|   |-- main_figure4.py
|   |-- main_figure5.py
|   `-- outputs/
|-- notebooks/
|   |-- figure_01.ipynb
|   `-- figure_02.ipynb
`-- results/
    |-- singlepop/
    |-- monopop_output/
    |-- twopop_output/
    |-- twopop_regime_samples/
    |-- clustering/
    |-- mfop_results/
    |-- spike_transmission/
    `-- te/
```

Generated binaries, object files, `__pycache__`, and large result artifacts are
not expanded in the tree above.

## Build

Compile a simulation source file with:

```sh
sh simulations/compile.sh <simulation_name>
```

Example:

```sh
sh simulations/compile.sh main_single
```

The script builds the shared C objects in `include/`, links against
`lib/libhhnet.a`, and writes the executable to
`simulations/<simulation_name>.out`.

## Simulation and Analysis Map

### Single-Population Simulations

`simulations/main_single.c`

- Simulates one hard-coded single population, either `pop_F` or `pop_S`.
- Use this for fixed-parameter examples and activity inspection.

`simulations/main_single_2d.c`

- Simulates a single population while varying `p_E` and `nu`.
- Requires MPI.
- Related analysis:
  - `analysis/export_singlepop_summary.py`: summarizes the simulation output.
  - `analysis/extract_burstprobs_singlepop.py`: extracts burst probability
    statistics from single-population activity.

### Connected Two-Population Simulations

`simulations/main_twopop.c`

- Simulates connected populations, typically `pop_F - pop_S`.
- Uses hard-coded parameters.

`simulations/main_twopop_4d.c`

- Simulates connected populations while varying `alpha`, `beta`, `Echelon`,
  and `omega`.
- Check the parameter-grid size around `int max_len[]` in the source before
  launching a sweep.
- Related setup and analysis:
  - `simulations/export_regime_params.py`: exports simulation parameters and
    should be run before the simulation sweep.
  - `analysis/export_twopop_summary.py`: summarizes two-population simulation
    results.
  - `analysis/run_kmeans_clustering.py`: runs KMeans clustering on oscillation
    features.
  - `analysis/clustering_oscillation_features.ipynb`: inspects clusters and
    selects landmarks.

`simulations/main_monopop_4d.c`

- Simulates connected homogeneous populations, such as `pop_F - pop_F` or
  `pop_S - pop_S`.
- Varies `alpha`, `beta`, `Echelon`, and positive `omega`.

### Regime Sampling and MFOP Analysis

`simulations/main_regime_samples.c`

- Samples activity from each dynamical regime.
- The current description assumes `nregimes = 7`.
- Related analysis:
  - `analysis/compute_coburst_probs.py`: computes co-burst probability maps.
  - `analysis/extract_burstprobs_twopop.py`: extracts burst probabilities using
    the filtration method.
  - `analysis/determine_frequency_range.ipynb`: determines MFOP frequency
    ranges using a Cauchy-distribution fit.
  - `analysis/identify_mfop.py`: identifies MFOPs. Run
    `determine_frequency_range.ipynb` first to set the frequency range for each
    dynamical regime.
  - `analysis/convert_mua.py`: converts spike signals into MUA signals.
  - `analysis/compute_te.py`: computes transfer entropy.

### Spike Transmission

`simulations/main_transmission.c`

- Simulates delayed spike transmission while varying delays.
- Related setup and analysis:
  - `simulations/export_transmission_params.py`: exports delayed spike
    transmission simulation parameters.
  - `analysis/identify_mfop_transmission.py`: identifies MFOPs in transmission
    simulations.
  - `analysis/compute_recv_response.py`: computes receiver-neuron response.
  - `analysis/compute_kappa.py`: computes kappa statistics for spike
    transmission outputs.

## Result Directories

- `results/singlepop/`: single-population sweep outputs and burst properties.
- `results/monopop_output/`: homogeneous connected-population summaries.
- `results/twopop_output/`: connected two-population summaries and processed
  outputs.
- `results/twopop_regime_samples/`: sampled regime activities, co-burst maps,
  and burst-probability outputs.
- `results/clustering/`: KMeans clustering outputs, cluster IDs, and landmarks.
- `results/mfop_results/`: MFOP motif results and diagnostic figures.
- `results/spike_transmission/`: receiver-response probabilities and kappa
  outputs.
- `results/te/`: transfer-entropy outputs.

## Figures

The `figures/main_figure*.py` scripts assemble manuscript-style figures from
the processed outputs in `results/`. Generated figure assets are stored under
`figures/outputs/`.
