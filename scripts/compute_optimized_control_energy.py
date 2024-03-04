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
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import normalize_state, matrix_normalization
from src.neural_network import train_nct

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
    time_horizon = config['time_horizon']
    c = config['c']
    rho = config['rho']

    # load A matrix
    adjacency = np.load(os.path.join(indir, A_file))
    system = 'continuous'
    adjacency_norm = matrix_normalization(adjacency, system=system, c=c)

    # load rsfMRI clusters
    fmri_clusters = np.load(os.path.join(outdir, fmri_clusters_file), allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    [n_states, n_nodes] = centroids.shape

    print('\nk={0}; time_horizon={1}; c={2}; rho={3}'.format(n_states, time_horizon, c, rho))
    control_set = np.eye(n_nodes)  # define control set using a uniform full control set
    trajectory_constraints = np.eye(n_nodes)  # define state trajectory constraints
    n_steps = 5  # number of gradient steps
    lr = 0.1  # learning rate for gradient

    start = time.time()
    # --- compute control energy. Reference state = target state ---
    print('Compute control energy. Reference state = target state...')
    optimized_weights_xf = np.zeros((n_states, n_states, n_steps, n_nodes))
    control_energy_xf = np.zeros((n_states, n_states))
    for initial_idx in tqdm(np.arange(n_states)):
        initial_state = normalize_state(centroids[initial_idx, :])  # initial state
        for target_idx in np.arange(n_states):
            if initial_idx != target_idx:
                target_state = normalize_state(centroids[target_idx, :])  # target state
                reference_state = target_state

                loss, opt_weights = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
                                              time_horizon=time_horizon, control_set=control_set,
                                              reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
                                              n_steps=n_steps, lr=lr)
                optimized_weights_xf[initial_idx, target_idx] = opt_weights

                adjacency_norm_opt = adjacency_norm - np.diag(opt_weights[-1, :])  # A_norm with optimized self-inhibition weights
                if np.any(np.diag(adjacency_norm_opt) > 0):
                    print('Found {0} positive diagonals on adjacency_norm_opt'.format(np.sum(np.diag(adjacency_norm_opt) > 0)))

                # get the state trajectory and the control signals
                state_trajectory, control_signals, numerical_error = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
                                                                                        T=time_horizon, B=control_set,
                                                                                        xr=reference_state, rho=rho, S=trajectory_constraints,
                                                                                        system=system)

                # integrate control signals to get control energy
                node_energy = integrate_u(control_signals)
                # summarize nodal energy
                control_energy_xf[initial_idx, target_idx] = np.sum(node_energy)

    # --- compute control energy. Reference state = midpoint ---
    print('Compute control energy. Reference state = midpoint...')
    optimized_weights_midpoint = np.zeros((n_states, n_states, n_steps, n_nodes))
    control_energy_midpoint = np.zeros((n_states, n_states))
    reference_state = 'midpoint'  # reference state
    for initial_idx in tqdm(np.arange(n_states)):
        initial_state = normalize_state(centroids[initial_idx, :])  # initial state
        for target_idx in np.arange(n_states):
            if initial_idx != target_idx:
                target_state = normalize_state(centroids[target_idx, :])  # target state

                loss, opt_weights = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
                                                    time_horizon=time_horizon, control_set=control_set,
                                                    reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
                                                    n_steps=n_steps, lr=lr)
                optimized_weights_midpoint[initial_idx, target_idx] = opt_weights

                adjacency_norm_opt = adjacency_norm - np.diag(opt_weights[-1, :])  # A_norm with optimized self-inhibition weights
                if np.any(np.diag(adjacency_norm_opt) > 0):
                    print('Found {0} positive diagonals on adjacency_norm_opt'.format(np.sum(np.diag(adjacency_norm_opt) > 0)))

                # get the state trajectory and the control signals
                state_trajectory, control_signals, numerical_error = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
                                                                                        T=time_horizon, B=control_set,
                                                                                        xr=reference_state, rho=rho, S=trajectory_constraints,
                                                                                        system=system)

                # integrate control signals to get control energy
                node_energy = integrate_u(control_signals)
                # summarize nodal energy
                control_energy_midpoint[initial_idx, target_idx] = np.sum(node_energy)

    # --- compute control energy. Reference state = zeros ---
    print('Compute control energy. Reference state = zeros...')
    optimized_weights_zero = np.zeros((n_states, n_states, n_steps, n_nodes))
    control_energy_zero = np.zeros((n_states, n_states))
    reference_state = 'zero'  # reference state
    for initial_idx in tqdm(np.arange(n_states)):
        initial_state = normalize_state(centroids[initial_idx, :])  # initial state
        for target_idx in np.arange(n_states):
            if initial_idx != target_idx:
                target_state = normalize_state(centroids[target_idx, :])  # target state

                loss, opt_weights = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
                                              time_horizon=time_horizon, control_set=control_set,
                                              reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
                                              n_steps=n_steps, lr=lr)
                optimized_weights_zero[initial_idx, target_idx] = opt_weights

                adjacency_norm_opt = adjacency_norm - np.diag(opt_weights[-1, :])  # A_norm with optimized self-inhibition weights
                if np.any(np.diag(adjacency_norm_opt) > 0):
                    print('Found {0} positive diagonals on adjacency_norm_opt'.format(np.sum(np.diag(adjacency_norm_opt) > 0)))

                # get the state trajectory and the control signals
                state_trajectory, control_signals, numerical_error = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
                                                                                        T=time_horizon, B=control_set,
                                                                                        xr=reference_state, rho=rho, S=trajectory_constraints,
                                                                                        system=system)

                # integrate control signals to get control energy
                node_energy = integrate_u(control_signals)
                # summarize nodal energy
                control_energy_zero[initial_idx, target_idx] = np.sum(node_energy)

    # --- compute control energy. Reference state = fMRI clusters ---
    # print('Compute control energy. Reference state = fMRI clusters...')
    # control_energy_ref = np.zeros((n_states, n_states, n_states))
    # optimized_weights = np.zeros((n_states, n_states, n_states, n_steps, n_nodes))
    # for initial_idx in tqdm(np.arange(n_states)):
    #     initial_state = normalize_state(centroids[initial_idx, :])  # initial state
    #     for target_idx in np.arange(n_states):
    #         if initial_idx != target_idx:
    #             target_state = normalize_state(centroids[target_idx, :])  # target state
    #             for reference_idx in np.arange(n_states):
    #                 reference_state = normalize_state(centroids[reference_idx, :])  # reference state
    #
    #                 loss, opt_weights = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
    #                                               time_horizon=time_horizon, control_set=control_set,
    #                                               reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
    #                                               n_steps=n_steps, lr=lr)
    #
    #                 adjacency_norm_opt = adjacency_norm - np.diag(opt_weights[-1, :])  # A_norm with optimized self-inhibition weights
    #                 # print(optimized_weights[-1, :5], np.diag(adjacency_norm_opt[:5, :5]))
    #
    #                 # get the state trajectory and the control signals
    #                 state_trajectory, control_signals, numerical_error = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
    #                                                                                         T=time_horizon, B=control_set,
    #                                                                                         xr=reference_state, rho=rho, S=trajectory_constraints,
    #                                                                                         system=system)
    #
    #                 # integrate control signals to get control energy
    #                 node_energy = integrate_u(control_signals)
    #                 # summarize nodal energy
    #                 control_energy_ref[initial_idx, target_idx, reference_idx] = np.sum(node_energy)
    #                 optimized_weights[initial_idx, target_idx, reference_idx] = opt_weights

    # save outputs
    log_args = {
        'control_energy_xf': control_energy_xf,
        'optimized_weights_xf': optimized_weights_xf,
        'control_energy_midpoint': control_energy_midpoint,
        'optimized_weights_midpoint': optimized_weights_midpoint,
        'control_energy_zero': control_energy_zero,
        'optimized_weights_zero': optimized_weights_zero,
        # 'control_energy_ref': control_energy_ref,
    }
    file_str = '{0}optimized_control_energy_k-{1}_T-{2}_c-{3}_rho-{4}'.format(file_prefix, n_states, time_horizon, c, rho)
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
    parser.add_argument('--time_horizon', type=int, default=1)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--rho', type=float, default=1)

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
        'time_horizon': args.time_horizon,
        'c': args.c,
        'rho': args.rho,
    }

    run(config=config)
