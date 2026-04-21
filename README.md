# Time-structured communication through cross-frequency bursts

This repository contains C-based Hodgkin-Huxley network simulations and Python
analysis pipelines for studying oscillatory regimes, multi-frequency oscillatory
patterns (MFOPs), transfer entropy, and delayed spike transmission in
single-population and coupled-population networks.

## Install

Install the local Python package in editable mode from the repository root:

```bash
> pip install -e .
```

## Build

Build the shared simulation library first:

```bash
> cd include
> make main
```

This creates `lib/libhhnet.a`. To compile an individual simulation program, run:

```bash
> sh simulations/compile.sh <simulation_name>
```

Example:

```bash
> sh simulations/compile.sh main_single # This will create simulations/main_single.out
```

The compile script builds the required objects in `include/`, links against
`lib/libhhnet.a`, and writes the executable to
`simulations/<simulation_name>.out`. MPI-enabled simulations require `mpicc`.

## Simulation and Analysis Map

### Single-Population Simulations

`simulations/main_single.c`

- Simulates a single population, either `pop_F` or `pop_S`, with hard-coded
  parameters.
- Intended for fixed-parameter examples and activity inspection.

`simulations/main_single_2d.c`

- Simulates a single population while varying `p_E` and `nu`.
- Requires MPI.
- Related analysis:
  - `analysis/export_singlepop_summary.py`: summarizes the simulation output.
  - `analysis/extract_burstprobs_singlepop.py`: extracts burst probability
    statistics from single-population activity.

### Connected Two-Population Simulations

`simulations/main_twopop.c`

- Simulates a pair of connected populations, typically `pop_F - pop_S`.
- Uses hard-coded parameters.

`simulations/main_twopop_4d.c`

- Simulates connected populations while varying `alpha`, `beta`, `rank` or
  echelon-like ordering, and `omega`.
- Check the parameter-grid size near `int max_len[]` before launching a sweep.
- Related setup and analysis:
  - `simulations/export_regime_params.py`: exports simulation parameters and
    should be run before the sweep.
  - `analysis/export_twopop_summary.py`: summarizes two-population simulation
    results.
  - `analysis/run_kmeans_clustering.py`: runs KMeans clustering on oscillation
    features.
  - `analysis/clustering_oscillation_features.ipynb`: inspects clusters and
    selects landmarks.

`simulations/main_monopop_4d.c`

- Simulates coupled homogeneous populations, such as `pop_F - pop_F` or
  `pop_S - pop_S`.
- Varies `alpha`, `beta`, `rank` or echelon-like ordering, and positive
  `omega`.

### Regime Sampling and MFOP Analysis

`simulations/main_regime_samples.c`

- Samples activity from each dynamical regime. The current setting assumes
  `nregimes = 7`.
- Related analysis:
  - `analysis/compute_coburst_probs.py`: computes co-burst probability maps.
  - `analysis/extract_burstprobs_twopop.py`: extracts burst probabilities using
    the filtration method.
  - `analysis/determine_frequency_range.ipynb`: determines MFOP frequency
    ranges using a Cauchy-distribution fit.
  - `analysis/identify_mfop.py`: identifies MFOPs. Run
    `analysis/determine_frequency_range.ipynb` first to set the frequency range
    for each dynamical regime.
  - `analysis/convert_mua.py`: converts spike signals into MUA signals.
  - `analysis/compute_te.py`: computes transfer entropy.

### Spike Transmission

`simulations/main_transmission.c`

- Simulates delayed spike transmission across different delay values.
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

The `figures/main_figure*.py` scripts assemble manuscript-style figures from the
processed outputs in `results/`. Generated figure assets are stored under
`figures/outputs/`.
