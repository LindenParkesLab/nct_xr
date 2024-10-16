import os, sys, time, getpass
import numpy as np

username = getpass.getuser()
if username == 'lindenmp':
    sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])
elif username == 'lp756':
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/nctpy/src'])

from nctpy.energies import get_control_inputs, integrate_u
from snaplab_tools.utils import get_schaefer_system_mask

def control_energy_helper(inputs):
    state_trajectory, control_signals, n_err = get_control_inputs(A_norm=inputs['adjacency_norm'], x0=inputs['initial_state'], xf=inputs['target_state'],
                                                                  T=inputs['time_horizon'], B=inputs['control_set'], xr=inputs['reference_state'],
                                                                  rho=inputs['rho'], S=inputs['trajectory_constraints'], system=inputs['system'])

    # integrate control signals to get control energy
    node_energy = integrate_u(control_signals)
    # summarize nodal energy
    control_energy = np.sum(node_energy)

    costate_error = n_err[0]
    xf_error = n_err[1]
 
    return state_trajectory, control_signals, node_energy, control_energy, costate_error, xf_error


def get_random_partial_control_set(n_nodes, n_control_nodes, add_small_control=False, seed=0):
    control_set = np.zeros((n_nodes, n_nodes))
    if add_small_control:
        control_set[np.eye(n_nodes) == 1] = 1e-5

    np.random.seed(seed)
    x = np.random.choice(np.arange(n_nodes), size=n_control_nodes, replace=False)
    for i in x:
        control_set[i, i] = 1
        
    return control_set


def get_yeo_control_set(node_labels, system, add_small_control=False):
    n_nodes = len(node_labels)
    control_set = np.zeros((n_nodes, n_nodes))
    if add_small_control:
        control_set[np.eye(n_nodes) == 1] = 1e-3

    system_mask = get_schaefer_system_mask(node_labels, system=system)
    for i in np.arange(n_nodes):
        if system_mask[i]:
            control_set[i, i] = 1
            
    return control_set


def get_adj_weights(log_args):
    n_states = log_args['optimized_weights'].shape[0]
    n_nodes = log_args['optimized_weights'].shape[-1]

    adjacency_weights = np.zeros((n_states, n_states, n_nodes))
    for initial_idx in np.arange(n_states):
        for target_idx in np.arange(n_states):
            try:
                idx = np.where(np.isnan(log_args['loss'][initial_idx, target_idx]))[0][0] - 1
            except:
                idx = log_args['loss'].shape[-1] - 1
            optimized_weights = -1 - log_args['optimized_weights'][initial_idx, target_idx, idx]
            if np.any(optimized_weights > 0):
                print('warning, positive weights found')
            adjacency_weights[initial_idx, target_idx] = optimized_weights
    
    return adjacency_weights
