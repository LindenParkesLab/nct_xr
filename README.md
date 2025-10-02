# nct_xr

[![DOI](https://zenodo.org/badge/762404607.svg)](https://doi.org/10.5281/zenodo.17253851)

This repository includes to replicate analyses in: [Kim et al., bioRxiv, 2025](https://www.biorxiv.org/content/10.1101/2025.04.24.650287v1)

In `scripts`:

- `computer_fmri_clusters.py`: This Python script takes in fMRI time series data and clusters it in time using *K*-means to derive empirical brain states for NCT analysis. The fMRI data is assumed to be stored in a `.npy` file structured as a `n_trs * n_nodes * n_scans * n_subs` array. Note, if more than one fMRI scan is stored on the 3rd axis, the code will take the first scan (e.g., `fmri_data = fmri_data[:, :, 0, :]`)
- `results_fmri_clusters.ipynb`: This Python notebook plots the brain states derived using `computer_fmri_clusters.py`. Note, you produce all plots, `computer_fmri_clusters.py` needs to be run multiple times, each time at a different *k*.
- `compute_optimized_control_energy.py`: This Python script takes in a set of brain states and a structural connectome, and runs our optimized NCT model. This script will output a set of optimized self-inhibition parameters, one per state transition.
- `results_optimized_control_energy.ipynb`: This Python notebook produces panels for Figure 2C and Figure 3.
- `results_optimized_weights.ipynb`: This Python notebook produces panels for Figure 4 and Figure 5.
- `results_optimized_weights_subjects.ipynb`: This Python notebook produces panels for Figure 7A.
- `results_control_energy_subjects.ipynb`: This Python notebook produces panels for Figure 7B.

In `src`:

- `neural_network.py`: Python code base for our algorithm, implemented in Pytorch

### Data

Note, this repository does not include the processed data used in the above manuscript. These data can be found elsewhere (see Data availability statement in the manuscript) and require further processing before results can be reproduced. 

For the Human Connectome Project data, minimally processed data can be downloaded from https://db.humanconnectome.org. Further processing of these data is required. Scripts to generate structural connectomes can be found in `scripts/processing/hcp_ya`. A list of subjects we used can be found in `scripts/processing/hcp_ya/HCPYA_Schaefer4007_subjids.txt`.
