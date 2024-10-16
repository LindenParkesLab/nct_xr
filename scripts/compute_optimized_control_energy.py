# import
import os, sys, warnings, argparse, time, getpass
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm

username = getpass.getuser()
if username == 'lindenmp':
    sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])
elif username == 'lp756':
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/nctpy/src'])

from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import normalize_state, matrix_normalization
from src.neural_network import train_nct
from src.utils import control_energy_helper, get_random_partial_control_set, get_yeo_control_set
from snaplab_tools.utils import get_schaefer_system_mask

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def run(config):
    outdir = config['outdir']
    A_file = config['A_file']
    fmri_clusters_file = config['fmri_clusters_file']

    # load rsfMRI clusters
    fmri_clusters = np.load(fmri_clusters_file, allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    if np.any(np.isnan(centroids)):
        print('WARNING: Found NaNs in centroids... filling within zeros')
        nan_mask = np.isnan(centroids)
        centroids[nan_mask] = 0
    [n_states, n_nodes] = centroids.shape
    print('\nbrain state params: k = {0}'.format(n_states))
    
    # nct params
    system = 'continuous'
    c = config['c']
    time_horizon = config['time_horizon']
    rho = config['rho']    
    control_set = np.eye(n_nodes)  # define control set using a uniform full control set
    if config['optimal_control']:
        trajectory_constraints = np.eye(n_nodes)  # define state trajectory constraints
    else:
        trajectory_constraints = np.zeros((n_nodes, n_nodes))  # define state trajectory constraints

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
    adjacency = np.load(A_file)
    file_prefix = config['file_prefix']
    perm_idx = config['perm_idx']
    if adjacency.ndim == 2:
        print('Found group averaged connectome')
        file_prefix = '{0}_'.format(file_prefix)
    elif adjacency.ndim == 3:
        print('Found {0} connectomes, using connectome {1}'.format(adjacency.shape[-1], perm_idx))
        file_prefix = '{0}-adj-{1}_'.format(file_prefix, perm_idx)

    # permute rsfMRI clusters
    permute_state = config['permute_state']
    if permute_state != False:
        try:
            fmri_clusters_file_permuted = config['fmri_clusters_file_permuted']
            states_permuted = np.load(fmri_clusters_file_permuted)
            print('Found {0} permuted states, using permutation {1}'.format(states_permuted.shape[-1], perm_idx))
            file_prefix = '{0}surr-{1}-{2}_'.format(file_prefix, permute_state, perm_idx)
        except:
            print('No permuted states found!')
            # print('No surrogate cluster file provided. Shuffling centroids...')
            # n_perms = 1000
            # states_permuted = np.zeros((n_states, n_nodes, n_perms))
            # for i in np.arange(n_states):
            #     for j in np.arange(n_perms):
            #         centroid = centroids[i, :].copy()
            #         np.random.seed(j)
            #         np.random.shuffle(centroid)
            #         states_permuted[i, :, j] = centroid.copy()

    # if 'A0' in file_prefix:
        # print('Nuking adj weights')
        # adjacency[:] = 0
    
    if config['optimal_control']:
        file_prefix += 'optimal-'
    else:
        file_prefix += 'minimum-'

    file_str = '{0}optimized-energy_k-{1}_c-{2}_T-{3}_rho-{4}_refstate-{5}_initweights-{6}_nsteps-{7}_lr-{8}_eigweight-{9}_regweight-{10}_regtype-{11}'.format(file_prefix, 
                                                                                                                                                               n_states, 
                                                                                                                                                               c, time_horizon, rho,
                                                                                                                                                               reference_state, init_weights,
                                                                                                                                                               n_steps, lr, eig_weight, reg_weight, reg_type)
    print(file_str)

    if config['outsubdir'] != '':
        outdir = os.path.join(outdir, config['outsubdir'])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print('saving to.. {0}'.format(outdir))
    if config['compact_save'] is True:
        print('\tusing compact save')

    if os.path.isfile(os.path.join(outdir, file_str + '.npy')):
        print('Output found. Skipping.')
    else:
        # normalize adjacency matrix
        if adjacency.ndim == 2:
            adjacency_norm = matrix_normalization(adjacency, system=system, c=c)
        elif adjacency.ndim == 3:
            adjacency_norm = matrix_normalization(adjacency[:, :, perm_idx], system=system, c=c)

        # compute control energy
        start = time.time()
        log_args = dict()
        loss = np.zeros((n_states, n_states, n_steps))
        eigen_values = np.zeros((n_states, n_states, n_steps))
        optimized_weights = np.zeros((n_states, n_states, n_steps, n_nodes))
        optimized_weights_final = np.zeros((n_states, n_states, n_nodes))

        n_timepoints = time_horizon * 1000 + 1
        state_trajectory = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        control_signals = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        numerical_error = np.zeros((n_states, n_states, 2))
        control_energy = np.zeros((n_states, n_states))

        state_trajectory_variable_decay = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        control_signals_variable_decay = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        numerical_error_variable_decay = np.zeros((n_states, n_states, 2))
        control_energy_variable_decay = np.zeros((n_states, n_states))

        state_trajectory_static_decay = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        control_signals_static_decay = np.zeros((n_states, n_states, n_timepoints, n_nodes))
        numerical_error_static_decay = np.zeros((n_states, n_states, 2))
        control_energy_static_decay = np.zeros((n_states, n_states))
        
        if config['run_rand_control_set'] is True:
            print('run_rand_control_set is True')
            n_control_nodes = np.linspace(5, n_nodes-5, 50).astype(int)
            n_unique_cns = n_control_nodes.shape[0]
            n_samples = 50
            print(n_unique_cns, n_control_nodes)
            
            control_signals_corr_partial = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            control_energy_partial = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            numerical_error_partial = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            xfcorr_partial = np.zeros((n_states, n_states, n_unique_cns, n_samples))

            control_signals_corr_partial_variable_decay = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            control_energy_partial_variable_decay = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            numerical_error_partial_variable_decay = np.zeros((n_states, n_states, n_unique_cns, n_samples))
            xfcorr_partial_variable_decay = np.zeros((n_states, n_states, n_unique_cns, n_samples))
        else:
            print('run_rand_control_set is False')
            
        if config['run_yeo_control_set'] is True:
            print('run_yeo_control_set is True')
            parc_centroids = pd.read_csv(config['parc_file'], index_col=0)
            yeo_systems = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
            n_systems = len(yeo_systems)

            control_signals_system = np.zeros((n_states, n_states, n_timepoints, n_nodes, n_systems))
            control_signals_corr_system = np.zeros((n_states, n_states, n_systems))
            control_energy_system = np.zeros((n_states, n_states, n_systems))
            numerical_error_system = np.zeros((n_states, n_states, n_systems))
            xfcorr_system = np.zeros((n_states, n_states, n_systems))

            control_signals_system_variable_decay = np.zeros((n_states, n_states, n_timepoints, n_nodes, n_systems))
            control_signals_corr_system_variable_decay = np.zeros((n_states, n_states, n_systems))
            control_energy_system_variable_decay = np.zeros((n_states, n_states, n_systems))
            numerical_error_system_variable_decay = np.zeros((n_states, n_states, n_systems))
            xfcorr_system_variable_decay = np.zeros((n_states, n_states, n_systems))
        else:
            print('run_yeo_control_set is False')

        for initial_idx in np.arange(n_states):
            if permute_state == 'initial':
                initial_state = normalize_state(states_permuted[initial_idx, :, perm_idx])  # initial state            
            else:
                initial_state = normalize_state(centroids[initial_idx, :])  # initial state
            
            for target_idx in np.arange(n_states):
                if permute_state == 'target':
                    target_state = normalize_state(states_permuted[target_idx, :, perm_idx])  # target state
                else:
                    target_state = normalize_state(centroids[target_idx, :])  # target state
                print('initial_state = {0}, target_state = {1}'.format(initial_idx, target_idx))
                
                if permute_state == 'midpoint':
                    ref_state = normalize_state(states_permuted[initial_idx, target_idx, :, perm_idx])
                else:
                    ref_state = reference_state
                
                ##############################################################################################################################################
                # setup inputs
                inputs = dict()
                inputs['adjacency_norm'] = adjacency_norm.copy()
                inputs['initial_state'] = initial_state.copy()
                inputs['target_state'] = target_state.copy()
                inputs['time_horizon'] = time_horizon
                inputs['control_set'] = control_set.copy()
                try:
                    inputs['reference_state'] = ref_state.copy()
                except:
                    inputs['reference_state'] = ref_state
                inputs['rho'] = rho
                inputs['trajectory_constraints'] = trajectory_constraints.copy()
                inputs['system'] = system
                
                ##############################################################################################################################################
                # 1) get bog standard optimal control energy
                state_trajectory[initial_idx, target_idx], control_signals[initial_idx, target_idx], _, control_energy[initial_idx, target_idx], \
                    numerical_error[initial_idx, target_idx, 0], numerical_error[initial_idx, target_idx, 1] = control_energy_helper(inputs=inputs)
                ##############################################################################################################################################

                ##############################################################################################################################################
                # 2) train model to get opimtizied decay rates
                loss[initial_idx, target_idx], \
                eigen_values[initial_idx, target_idx], \
                optimized_weights[initial_idx, target_idx] = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state, 
                                                                       time_horizon=time_horizon, control_set=control_set, reference_state=ref_state, 
                                                                       rho=rho, trajectory_constraints=trajectory_constraints, init_weights=init_weights,
                                                                       n_steps=n_steps, lr=lr, eig_weight=eig_weight, reg_weight=reg_weight, reg_type=reg_type,
                                                                       early_stopping=early_stopping)
                try:
                    idx = np.where(np.isnan(loss[initial_idx, target_idx]))[0][0] - 1
                except:
                    idx = n_steps - 1
                optimized_weights_final[initial_idx, target_idx, :] = optimized_weights[initial_idx, target_idx, idx, :].copy()
                adjacency_weights = -1 - optimized_weights_final[initial_idx, target_idx, :]
                if np.any(adjacency_weights > 0):
                    print('warning, positive weights found')
                ##############################################################################################################################################

                ##############################################################################################################################################
                # 3) get the state trajectory and the control signals with VARIABLE optimized self-inhibition weights
                adjacency_norm_opt = adjacency_norm.copy()
                adjacency_norm_opt[np.eye(n_nodes) == 1] = adjacency_weights
                inputs['adjacency_norm'] = adjacency_norm_opt.copy()

                state_trajectory_variable_decay[initial_idx, target_idx], control_signals_variable_decay[initial_idx, target_idx], _, control_energy_variable_decay[initial_idx, target_idx], \
                    numerical_error_variable_decay[initial_idx, target_idx, 0], numerical_error_variable_decay[initial_idx, target_idx, 1] = control_energy_helper(inputs=inputs)
                ##############################################################################################################################################

                ##############################################################################################################################################
                # 4) get the state trajectory and the control signals with STATIC optimized self-inhibition weights
                adjacency_norm_opt = adjacency_norm.copy()
                adjacency_norm_opt[np.eye(n_nodes) == 1] = np.mean(adjacency_weights)
                inputs['adjacency_norm'] = adjacency_norm_opt.copy()

                state_trajectory_static_decay[initial_idx, target_idx], control_signals_static_decay[initial_idx, target_idx], _, control_energy_static_decay[initial_idx, target_idx], \
                    numerical_error_static_decay[initial_idx, target_idx, 0], numerical_error_static_decay[initial_idx, target_idx, 1] = control_energy_helper(inputs=inputs)
                ##############################################################################################################################################

                # random partial control sets
                if config['run_rand_control_set'] is True:
                    print('\trunning random partial control sets...')

                    for i in tqdm(np.arange(n_unique_cns)):
                        for j in np.arange(n_samples):
                            inputs['control_set'] = get_random_partial_control_set(n_nodes=n_nodes, n_control_nodes=n_control_nodes[i], add_small_control=False, seed=j)

                            ##############################################################################################################################################
                            inputs['adjacency_norm'] = adjacency_norm.copy()
                            
                            state_traj, control_sig, _, control_energy_partial[initial_idx, target_idx, i, j], \
                                _, numerical_error_partial[initial_idx, target_idx, i, j] = control_energy_helper(inputs=inputs)
                            
                            xfcorr_partial[initial_idx, target_idx, i, j] = sp.stats.pearsonr(state_traj[-1, :], target_state)[0]
                            control_signals_corr_partial[initial_idx, target_idx, i, j] = np.nanmean(np.abs(np.corrcoef(control_sig.T))[np.triu_indices(n_nodes, k=1)])
                            ##############################################################################################################################################

                            ##############################################################################################################################################
                            adjacency_norm_opt = adjacency_norm.copy()
                            adjacency_norm_opt[np.eye(n_nodes) == 1] = adjacency_weights  
                            inputs['adjacency_norm'] = adjacency_norm_opt.copy()
                            
                            state_traj, control_sig, _, control_energy_partial_variable_decay[initial_idx, target_idx, i, j], \
                                _, numerical_error_partial_variable_decay[initial_idx, target_idx, i, j] = control_energy_helper(inputs=inputs)
                            
                            xfcorr_partial_variable_decay[initial_idx, target_idx, i, j] = sp.stats.pearsonr(state_traj[-1, :], target_state)[0]
                            control_signals_corr_partial_variable_decay[initial_idx, target_idx, i, j] = np.nanmean(np.abs(np.corrcoef(control_sig.T))[np.triu_indices(n_nodes, k=1)])
                            ##############################################################################################################################################
                            
                # random partial control sets
                if config['run_yeo_control_set'] is True:
                    print('\trunning yeo system partial control sets...')

                    for i in tqdm(np.arange(n_systems)):
                        control_set_yeo = get_yeo_control_set(list(parc_centroids.index), yeo_systems[i], add_small_control=True)
                        inputs['control_set'] = control_set_yeo.copy()
                        n_corr_vars = np.sum(control_set_yeo == 1)

                        ##############################################################################################################################################
                        inputs['adjacency_norm'] = adjacency_norm.copy()
                        
                        state_traj, control_sig, _, control_energy_system[initial_idx, target_idx, i], \
                            _, numerical_error_system[initial_idx, target_idx, i] = control_energy_helper(inputs=inputs)
                        
                        xfcorr_system[initial_idx, target_idx, i] = sp.stats.pearsonr(state_traj[-1, :], target_state)[0]
                        control_signals_system[initial_idx, target_idx, :, :, i] = control_sig.copy()
                        control_sig = control_sig[:, np.diag(control_set_yeo == 1)]
                        control_signals_corr_system[initial_idx, target_idx, i] = np.nanmean(np.abs(np.corrcoef(control_sig.T))[np.triu_indices(n_corr_vars, k=1)])
                        ##############################################################################################################################################

                        ##############################################################################################################################################
                        adjacency_norm_opt = adjacency_norm.copy()
                        adjacency_norm_opt[np.eye(n_nodes) == 1] = adjacency_weights  
                        inputs['adjacency_norm'] = adjacency_norm_opt.copy()
                        
                        state_traj, control_sig, _, control_energy_system_variable_decay[initial_idx, target_idx, i], \
                            _, numerical_error_system_variable_decay[initial_idx, target_idx, i] = control_energy_helper(inputs=inputs)
                        
                        xfcorr_system_variable_decay[initial_idx, target_idx, i] = sp.stats.pearsonr(state_traj[-1, :], target_state)[0]
                        control_signals_system_variable_decay[initial_idx, target_idx, :, :, i] = control_sig.copy()
                        control_sig = control_sig[:, np.diag(control_set_yeo == 1)]
                        control_signals_corr_system_variable_decay[initial_idx, target_idx, i] = np.nanmean(np.abs(np.corrcoef(control_sig.T))[np.triu_indices(n_corr_vars, k=1)])
                        ##############################################################################################################################################

        # save outputs
        if config['compact_save'] is True:
            # log_args['optimized_weights'] = optimized_weights
            log_args['optimized_weights'] = optimized_weights_final
            log_args['control_energy'] = control_energy
            log_args['control_energy_variable_decay'] = control_energy_variable_decay
            log_args['control_energy_static_decay'] = control_energy_static_decay
        else:
            log_args['state_trajectory'] = state_trajectory
            log_args['control_signals'] = control_signals
            log_args['numerical_error'] = numerical_error
            log_args['control_energy'] = control_energy    
            log_args['loss'] = loss
            log_args['eigen_values'] = eigen_values
            log_args['optimized_weights'] = optimized_weights

            log_args['state_trajectory_variable_decay'] = state_trajectory_variable_decay
            log_args['control_signals_variable_decay'] = control_signals_variable_decay
            log_args['numerical_error_variable_decay'] = numerical_error_variable_decay
            log_args['control_energy_variable_decay'] = control_energy_variable_decay

            log_args['state_trajectory_static_decay'] = state_trajectory_static_decay
            log_args['control_signals_variable_decay'] = control_signals_static_decay
            log_args['numerical_error_static_decay'] = numerical_error_static_decay
            log_args['control_energy_static_decay'] = control_energy_static_decay
            
            if config['run_rand_control_set'] is True:
                log_args['control_signals_corr_partial'] = control_signals_corr_partial
                log_args['control_energy_partial'] = control_energy_partial
                log_args['numerical_error_partial'] = numerical_error_partial
                log_args['xfcorr_partial'] = xfcorr_partial

                log_args['control_signals_corr_partial_variable_decay'] = control_signals_corr_partial_variable_decay
                log_args['control_energy_partial_variable_decay'] = control_energy_partial_variable_decay
                log_args['numerical_error_partial_variable_decay'] = numerical_error_partial_variable_decay
                log_args['xfcorr_partial_variable_decay'] = xfcorr_partial_variable_decay
                
            if config['run_yeo_control_set'] is True:
                log_args['control_signals_system'] = control_signals_system
                log_args['control_signals_corr_system'] = control_signals_corr_system
                log_args['control_energy_system'] = control_energy_system
                log_args['numerical_error_system'] = numerical_error_system
                log_args['xfcorr_system'] = xfcorr_system

                log_args['control_signals_system_variable_decay'] = control_signals_system_variable_decay
                log_args['control_signals_corr_system_variable_decay'] = control_signals_corr_system_variable_decay
                log_args['control_energy_system_variable_decay'] = control_energy_system_variable_decay
                log_args['numerical_error_system_variable_decay'] = numerical_error_system_variable_decay
                log_args['xfcorr_system_variable_decay'] = xfcorr_system_variable_decay

        np.save(os.path.join(outdir, file_str), log_args)

        end = time.time()
        print('...done in {:.2f} seconds.'.format(end - start))
        print('\n')

# %%
def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--A_file', type=str, default='/home/lindenmp/research_projects/nct_xr/data/hcp_schaefer400-7_Am-features_schaefer_streamcount_areanorm_log.npy')
    parser.add_argument('--fmri_clusters_file', type=str, default='/home/lindenmp/research_projects/nct_xr/results/hcp_fmri_clusters_k-7.npy')
    parser.add_argument('--fmri_clusters_file_permuted', type=str, default=None)
    # parser.add_argument('--fmri_clusters_file_permuted', type=str, default='/home/lindenmp/research_projects/nct_xr/results/hcp_fmri_clusters_k-7_brainsmash-surrogates-5000.npy')
    # parser.add_argument('--fmri_clusters_file_permuted', type=str, default='/home/lindenmp/research_projects/nct_xr/results/hcp_fmri_clusters_k-7_brainsmash-surrogates-reference-5000.npy')

    parser.add_argument('--outdir', type=str, default='/home/lindenmp/research_projects/nct_xr/results')
    parser.add_argument('--outsubdir', type=str, default='')

    parser.add_argument('--file_prefix', type=str, default='hcp-Am')
    parser.add_argument('--perm_idx', type=int, default=0)

    # settings
    parser.add_argument('--optimal_control', type=str, default='True')
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--time_horizon', type=int, default=1)
    parser.add_argument('--rho', type=float, default=1)
    
    parser.add_argument('--reference_state', type=str, default='xf')
    parser.add_argument('--permute_state', type=str, default='False')    
    parser.add_argument('--init_weights', type=str, default='one')
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eig_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=0.0001)
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--early_stopping', type=str, default='True')
    
    parser.add_argument('--run_rand_control_set', type=str, default='False')
    parser.add_argument('--run_yeo_control_set', type=str, default='False')
    parser.add_argument('--parc_file', type=str, default='/home/lindenmp/research_projects/nct_xr/data/schaefer400-7_centroids.csv')

    # save out options
    parser.add_argument('--compact_save', type=str, default='False')

    args = parser.parse_args()
    args.outdir = os.path.expanduser(args.outdir)

    return args

# %%
if __name__ == '__main__':
    args = get_args()

    if args.optimal_control == 'False':
        args.optimal_control = False
    elif args.optimal_control == 'True':
        args.optimal_control = True

    if args.early_stopping == 'False':
        args.early_stopping = False
    elif args.early_stopping == 'True':
        args.early_stopping = True
        
    if args.permute_state == 'False':
        args.permute_state = False

    if args.run_rand_control_set == 'False':
        args.run_rand_control_set = False
    elif args.run_rand_control_set == 'True':
        args.run_rand_control_set = True
        
    if args.run_yeo_control_set == 'False':
        args.run_yeo_control_set = False
    elif args.run_yeo_control_set == 'True':
        args.run_yeo_control_set = True

    if args.compact_save == 'False':
        args.compact_save = False
    elif args.compact_save == 'True':
        args.compact_save = True

    config = {
        'outdir': args.outdir,
        'outsubdir': args.outsubdir,
        'A_file': args.A_file,
        'fmri_clusters_file': args.fmri_clusters_file,
        'fmri_clusters_file_permuted': args.fmri_clusters_file_permuted,
        'file_prefix': args.file_prefix,
        'perm_idx': args.perm_idx,

        # settings
        'optimal_control': args.optimal_control,
        'c': args.c,
        'time_horizon': args.time_horizon,
        'rho': args.rho,

        'reference_state': args.reference_state,
        'permute_state': args.permute_state,
        'init_weights': args.init_weights,
        'n_steps': args.n_steps,
        'lr': args.lr,
        'eig_weight': args.eig_weight,
        'reg_weight': args.reg_weight,
        'reg_type': args.reg_type,
        'early_stopping': args.early_stopping,
        
        'run_rand_control_set': args.run_rand_control_set,
        'run_yeo_control_set': args.run_yeo_control_set,
        'parc_file': args.parc_file,

        'compact_save': args.compact_save,
    }

    run(config=config)
