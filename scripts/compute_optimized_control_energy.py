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

    # load rsfMRI clusters
    fmri_clusters = np.load(os.path.join(outdir, fmri_clusters_file), allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    [n_states, n_nodes] = centroids.shape
    print('\nbrain state params: k = {0}'.format(n_states))

    # nct params
    system = 'continuous'
    c = config['c']
    time_horizon = config['time_horizon']
    rho = config['rho']    
    control_set = np.eye(n_nodes)  # define control set using a uniform full control set
    trajectory_constraints = np.eye(n_nodes)  # define state trajectory constraints
    print('Computing control energy.')
    print('\tnct params: c = {0}; time_horizon = {1}; rho = {2}'.format(c, time_horizon, rho))

    # training params
    reference_state = config['reference_state']
    init_weights = config['init_weights']
    n_steps = config['n_steps']  # number of gradient steps
    lr = config['lr']  # learning rate for gradient
    eig_weight = config['eig_weight']  # regularization strength for eigen value penalty
    reg_weight = config['reg_weight']  # regularization strength for weight penalty (e.g., l2)
    reg_type = config['reg_type']  # regularization type: l1 or l2
    early_stopping = config['early_stopping']
    print('\ttraining params: reference_state = {0}; init_weights = {1}; n_steps = {2}; lr = {3}; eig_weight = {4}; reg_weight = {5}; reg_type = {6}'.format(reference_state, init_weights, n_steps,
                                                                                                                                                             lr, eig_weight, reg_weight, reg_type))

    # load A matrix
    adjacency = np.load(os.path.join(indir, A_file))
    if adjacency.ndim == 2:
        print('Found group averaged connectome')
        file_prefix = '{0}_'.format(config['file_prefix'])
    elif adjacency.ndim == 3:
        print('Found subject level connectomes')
        subj_idx = config['subj_idx']
        file_prefix = '{0}-subj-{1}_'.format(config['file_prefix'], subj_idx)

    file_str = '{0}optimized_control_energy_k-{1}_c-{2}_T-{3}_rho-{4}_refstate-{5}_initweights-{6}_nsteps-{7}_lr-{8}_eigweight-{9}_regweight-{10}_regtype-{11}'.format(file_prefix, 
                                                                                                                                                                       n_states, 
                                                                                                                                                                       c, time_horizon, rho,
                                                                                                                                                                       reference_state, init_weights,
                                                                                                                                                                       n_steps, lr, eig_weight, reg_weight, reg_type)
    print(file_str)

    if os.path.isfile(os.path.join(outdir, file_str + '.npy')):
        print('Output found. Skipping.')
    else:
        # normalize adjacency matrix
        if adjacency.ndim == 2:
            adjacency_norm = matrix_normalization(adjacency, system=system, c=c)
        elif adjacency.ndim == 3:
            adjacency_norm = matrix_normalization(adjacency[:, :, subj_idx], system=system, c=c)

        # compute control energy
        start = time.time()
        log_args = dict()
        loss = np.zeros((n_states, n_states, n_steps))
        eigen_values = np.zeros((n_states, n_states, n_steps))
        optimized_weights = np.zeros((n_states, n_states, n_steps, n_nodes))
        control_energy_variable_decay = np.zeros((n_states, n_states))
        control_energy_static_decay = np.zeros((n_states, n_states))

        for initial_idx in np.arange(n_states):
            initial_state = normalize_state(centroids[initial_idx, :])  # initial state
            for target_idx in np.arange(n_states):
                print('initial_state = {0}, target_state = {1}'.format(initial_idx, target_idx))
                target_state = normalize_state(centroids[target_idx, :])  # target state

                loss[initial_idx, target_idx], \
                eigen_values[initial_idx, target_idx], \
                optimized_weights[initial_idx, target_idx] = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state, time_horizon=time_horizon, control_set=control_set,
                                                                       reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints, init_weights=init_weights,
                                                                       n_steps=n_steps, lr=lr, eig_weight=eig_weight, reg_weight=reg_weight, reg_type=reg_type, early_stopping=early_stopping)
                try:
                    idx = np.where(np.isnan(loss[initial_idx, target_idx]))[0][0] - 1
                except:
                    idx = n_steps - 1
                adjacency_weights = -1 - optimized_weights[initial_idx, target_idx, idx, :]
                if np.any(adjacency_weights > 0):
                    print('warning, positive weights found')
                            
                # get the state trajectory and the control signals with VARIABLE optimized self-inhibition weights
                adjacency_norm_opt = adjacency_norm.copy()
                adjacency_norm_opt[np.eye(n_nodes) == 1] = adjacency_weights
                _, control_signals, _ = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
                                                           T=time_horizon, B=control_set, xr=reference_state, rho=rho, S=trajectory_constraints, system=system)
                # integrate control signals to get control energy
                node_energy = integrate_u(control_signals)
                # summarize nodal energy
                control_energy_variable_decay[initial_idx, target_idx] = np.sum(node_energy)
                
                # get the state trajectory and the control signals with STATIC optimized self-inhibition weights
                adjacency_norm_opt = adjacency_norm.copy()
                adjacency_norm_opt[np.eye(n_nodes) == 1] = np.mean(adjacency_weights)
                _, control_signals, _ = get_control_inputs(A_norm=adjacency_norm_opt, x0=initial_state, xf=target_state,
                                                           T=time_horizon, B=control_set, xr=reference_state, rho=rho, S=trajectory_constraints, system=system)
                # integrate control signals to get control energy
                node_energy = integrate_u(control_signals)
                # summarize nodal energy
                control_energy_static_decay[initial_idx, target_idx] = np.sum(node_energy)
        
            log_args['loss'] = loss
            log_args['eigen_values'] = eigen_values
            log_args['optimized_weights'] = optimized_weights
            log_args['control_energy_variable_decay'] = control_energy_variable_decay
            log_args['control_energy_static_decay'] = control_energy_static_decay

        # save outputs
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
    parser.add_argument('--A_file', type=str, default='hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy')
    parser.add_argument('--fmri_clusters_file', type=str, default='hcp_fmri_clusters_k-7.npy')
    parser.add_argument('--file_prefix', type=str, default='hcp')
    parser.add_argument('--subj_idx', type=int, default=0)

    # settings
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--time_horizon', type=int, default=1)
    parser.add_argument('--rho', type=float, default=1)
    
    parser.add_argument('--reference_state', type=str, default='midpoint')
    parser.add_argument('--init_weights', type=str, default='one')
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eig_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=0.0001)
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--early_stopping', type=bool, default=True)

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
        'subj_idx': args.subj_idx,

        # settings
        'c': args.c,
        'time_horizon': args.time_horizon,
        'rho': args.rho,

        'reference_state': args.reference_state,
        'init_weights': args.init_weights,
        'n_steps': args.n_steps,
        'lr': args.lr,
        'eig_weight': args.eig_weight,
        'reg_weight': args.reg_weight,
        'reg_type': args.reg_type,
        'early_stopping': args.early_stopping,
    }

    run(config=config)
