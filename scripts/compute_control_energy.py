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

    # load A matrix
    adjacency = np.load(os.path.join(indir, A_file))

    # load rsfMRI clusters
    fmri_clusters = np.load(os.path.join(outdir, fmri_clusters_file), allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    [n_states, n_nodes] = centroids.shape
    print('\nbrain state params: k = {0}'.format(n_states))

    # nct params
    system = 'continuous'
    optimal_control = config['optimal_control']
    c = config['c']
    time_horizon = config['time_horizon']
    rho = config['rho']    
    control_set = np.eye(n_nodes)  # define control set using a uniform full control set
    if optimal_control:
        trajectory_constraints = np.eye(n_nodes)  # define state trajectory constraints
    else:
        trajectory_constraints = np.zeros((n_nodes, n_nodes))  # define state trajectory constraints
    print('nct params: optimal control = {0}; c = {1}; time_horizon = {2}; rho = {3}'.format(optimal_control, c, time_horizon, rho))

    # compute control energy
    start = time.time()
    log_args = dict()

    # assemble control tasks
    control_tasks = []
    for initial_idx in np.arange(n_states):
        initial_state = normalize_state(centroids[initial_idx, :])  # initial state
        for target_idx in np.arange(n_states):
            target_state = normalize_state(centroids[target_idx, :])  # target state

            control_task = dict()  # initialize dict
            control_task["x0"] = initial_state  # store initial state
            control_task["xf"] = target_state  # store target state
            control_task["B"] = control_set  # store control set
            control_task["S"] = trajectory_constraints  # store state trajectory constraints
            control_task["rho"] = rho  # store rho
            control_tasks.append(control_task)

    for reference_state in ['zero', 'midpoint', 'xf']:
        print('\ncomputing control energy. Reference state = {0}...'.format(reference_state))

        for i in np.arange(len(control_tasks)):
            control_tasks[i]["xr"] = reference_state

        # compute control energy across all control tasks
        compute_control_energy = ComputeControlEnergy(A=adjacency,
                                                      control_tasks=control_tasks,
                                                      system=system, c=c, T=time_horizon)
        compute_control_energy.run()

        # reshape energy into matrix
        control_energy = np.reshape(compute_control_energy.E, (n_states, n_states))

        log_args['control_energy_{0}'.format(reference_state)] = control_energy

    # save outputs
    if optimal_control:
        file_str = '{0}optimal_control_energy_k-{1}_c-{2}_T-{3}_rho-{4}'.format(file_prefix, n_states, c, time_horizon, rho)
    else:
        file_str = '{0}minimum_control_energy_k-{1}_c-{2}_T-{3}_rho-{4}'.format(file_prefix, n_states, c, time_horizon, rho)
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

    parser.add_argument('--indir', type=str, default='/home/lindenmp/research_projects/nct_xr/data')
    parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    parser.add_argument('--A_file', type=str, default='hcp_schaefer400-7_Am.npy')
    parser.add_argument('--fmri_clusters_file', type=str, default='hcp_fmri_clusters_k-7.npy')
    parser.add_argument('--file_prefix', type=str, default='hcp_')

    # settings
    parser.add_argument('--optimal_control', type=bool, default=True)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--time_horizon', type=int, default=1)
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
        'optimal_control': args.optimal_control,
        'c': args.c,
        'time_horizon': args.time_horizon,
        'rho': args.rho,
    }

    run(config=config)
