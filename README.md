# nct_xr

This repository includes to replicate analyses in: [Kim et al., bioRxiv, 2025](https://www.biorxiv.org/content/10.1101/2025.04.24.650287v1)

In `scripts`:

- `computer_fmri_clusters.py`: This Python script takes in fMRI time series data and clusters it in time using *K*-means to derive empirical brain states for NCT analysis. The fMRI data is assumed to be stored in a `.npy` file structured as a `n_trs * n_nodes * n_scans * n_subs` array. Note, if more than one fMRI scan is stored on the 3rd axis, the code will take the first scan (e.g., `fmri_data = fmri_data[:, :, 0, :]`)
- `results_fmri_clusters.ipynb`: This Python notebook plots the brain states derived using `computer_fmri_clusters.py`. Note, you produce all plots, `computer_fmri_clusters.py` needs to be run multiple times, each time at a different *k*.
- `compute_optimized_control_energy.py`: This Python script takes in a set of brain states and a structural connectome, and runs our optimized NCT model. This script will output a set of optimized self-inhibition parameters, one per state transition.
- `results_optimized_control_energy.ipynb`: This Python notebook produces panels for Figure 2C and Figure 3.
- `results_control_energy_subjects.ipynb`: This Python notebook produces panels for Figure 7.

In `src`:

- `neural_network.py`: Python code base for our algorithm, implemented in Pytorch
