# import
import os, sys, warnings, argparse, time
import numpy as np
import scipy as sp
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from nilearn import datasets

sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])

# import plotting libraries
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10})
plt.rcParams["svg.fonttype"] = "none"
import seaborn as sns
from snaplab_tools.plotting.utils import set_plotting_params

from nctpy.pipelines import ComputeControlEnergy, ComputeOptimizedControlEnergy
from nctpy.utils import normalize_state

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def run(config):
    set_plotting_params()
    indir = config['indir']
    outdir = config['outdir']
    A_file = config['A_file']
    fmri_clusters_file = config['fmri_clusters_file']
    file_prefix = config['file_prefix']
    T = config['T']
    c = config['c']
    rho = config['rho']

    # load A matrix
    adjacency = np.load(os.path.join(indir, A_file))

    # load rsfMRI clusters
    fmri_clusters = np.load(os.path.join(outdir, fmri_clusters_file), allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    [n_states, n_nodes] = centroids.shape

    print('\nk={0}; T={1}; c={2}; rho={3}'.format(n_states, T, c, rho))

    start = time.time()
    # --- compute control energy. Reference state = target state ---
    print('Compute control energy. Reference state = target state...')
    # define control set using a uniform full control set
    control_set = np.eye(n_nodes)
    # define state trajectory constraints
    trajectory_constraints = np.eye(n_nodes)
    # assemble control tasks
    control_tasks = []
    for initial_idx in np.arange(n_states):
        initial_state = normalize_state(centroids[initial_idx, :])  # initial state
        for target_idx in np.arange(n_states):
            target_state = normalize_state(centroids[target_idx, :])  # target state

            control_task = dict()  # initialize dict
            control_task["x0"] = initial_state  # store initial state
            control_task["xf"] = target_state  # store target state
            control_task["xr"] = target_state  # reference state
            control_task["B"] = control_set  # store control set
            control_task["S"] = trajectory_constraints  # store state trajectory constraints
            control_task["rho"] = rho  # store rho
            control_tasks.append(control_task)

    # compute control energy across all control tasks
    compute_control_energy = ComputeControlEnergy(A=adjacency,
                                                  control_tasks=control_tasks,
                                                  system="continuous", c=c, T=T)
    compute_control_energy.run()

    # reshape energy into matrix
    control_energy_xf = np.reshape(compute_control_energy.E, (n_states, n_states))

    # --- compute control energy. Reference state = midpoint ---
    print('Compute control energy. Reference state = midpoint...')
    for i in np.arange(len(control_tasks)):
        control_tasks[i]["xr"] = control_tasks[i]["x0"] + ((control_tasks[i]["xf"] - control_tasks[i]["x0"]) * 0.5)  # reference state

    # compute control energy across all control tasks
    compute_control_energy = ComputeControlEnergy(A=adjacency,
                                                  control_tasks=control_tasks,
                                                  system="continuous", c=c, T=T)
    compute_control_energy.run()

    # reshape energy into matrix
    control_energy_midpoint = np.reshape(compute_control_energy.E, (n_states, n_states))

    # --- compute control energy. Reference state = zeros ---
    print('Compute control energy. Reference state = zeros...')
    for i in np.arange(len(control_tasks)):
        control_tasks[i]["xr"] = 'zero'  # reference state

    # compute control energy across all control tasks
    compute_control_energy = ComputeControlEnergy(A=adjacency,
                                                  control_tasks=control_tasks,
                                                  system="continuous", c=c, T=T)
    compute_control_energy.run()

    # reshape energy into matrix
    control_energy_zero = np.reshape(compute_control_energy.E, (n_states, n_states))

    # --- compute control energy. Reference state = fMRI clusters ---
    # print('Compute control energy. Reference state = fMRI clusters...')
    # control_energy_ref = np.zeros((n_states, n_states, n_states))
    # for ref_state in np.arange(n_states):
    #     reference_state = centroids[ref_state, :]
    #     reference_state = normalize_state(reference_state)
    #
    #     for i in np.arange(len(control_tasks)):
    #         control_tasks[i]["xr"] = reference_state  # reference state
    #
    #     # compute control energy across all control tasks
    #     compute_control_energy = ComputeControlEnergy(A=adjacency,
    #                                                   control_tasks=control_tasks,
    #                                                   system="continuous", c=c, T=T)
    #     compute_control_energy.run()
    #
    #     # reshape energy into matrix
    #     control_energy_ref[:, :, ref_state] = np.reshape(compute_control_energy.E, (n_states, n_states))

    # save outputs
    log_args = {
        'control_energy_xf': control_energy_xf,
        'control_energy_midpoint': control_energy_midpoint,
        'control_energy_zero': control_energy_zero,
        # 'control_energy_ref': control_energy_ref,
    }
    file_str = '{0}control_energy_k-{1}_T-{2}_c-{3}_rho-{4}'.format(file_prefix, n_states, T, c, rho)
    np.save(os.path.join(outdir, file_str), log_args)

    end = time.time()
    print('...done in {:.2f} seconds.'.format(end - start))

# %%
def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='/media/lindenmp/storage_ssd/research_projects/nct_xr/data')
    parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    parser.add_argument('--A_file', type=str, default='hcp_schaefer400-7_Am.npy')
    parser.add_argument('--fmri_clusters_file', type=str, default='hcp_fmri_clusters_k-10.npy')
    parser.add_argument('--file_prefix', type=str, default='hcp_')
    # parser.add_argument('--indir', type=str, default='/media/lindenmp/storage_ssd/research_projects/nct_xr/data/probabilistic_sift_radius2_count')
    # parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    # parser.add_argument('--A_file', type=str, default='pnc_schaefer400-7_Am.npy')
    # parser.add_argument('--fmri_clusters_file', type=str, default='pnc_fmri_clusters_k-5.npy')
    # parser.add_argument('--file_prefix', type=str, default='pnc_')

    # settings
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--rho', type=float, default=0.1)

    args = parser.parse_args()
    args.indir = os.path.expanduser(args.indir)
    args.outdir = os.path.expanduser(args.outdir)

    return args

# %%
if __name__ == '__main__':
    args = get_args()

    config = {
        'indir': args.indir,
        'outdir': args.outdir,
        'A_file': args.A_file,
        'fmri_clusters_file': args.fmri_clusters_file,
        'file_prefix': args.file_prefix,

        # settings
        'T': args.T,
        'c': args.c,
        'rho': args.rho,
    }

    run(config=config)
