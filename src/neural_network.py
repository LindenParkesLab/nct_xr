import os, sys, time, getpass
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

username = getpass.getuser()
if username == 'lindenmp':
    sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])
elif username == 'lp756':
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/nctpy/src'])

from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import normalize_state, matrix_normalization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
if device == 'cuda':
    cuda_device = torch.cuda.get_device_name(0)
    print(cuda_device)
    
    if 'Quadro' in cuda_device:
        device = 'cpu'
        print(device)

class NCT(nn.Module):
    def __init__(self, adjacency_norm, 
                 initial_state, target_state,
                 time_horizon, control_set,
                 reference_state, rho, trajectory_constraints,
                 init_weights='one',
                 eig_weight=0.1, reg_weight=0.0001, reg_type='l2'):
        super().__init__()
        n_nodes = adjacency_norm.shape[0]

        # state vectors to float if they're bools
        if type(initial_state[0]) == np.bool_:
            initial_state = initial_state.astype(float)
        if type(target_state[0]) == np.bool_:
            target_state = target_state.astype(float)

        # check dimensions of states
        if initial_state.ndim == 1:
            initial_state = initial_state[:, np.newaxis]
        if target_state.ndim == 1:
            target_state = target_state[:, np.newaxis]
        if reference_state.ndim == 1:
            reference_state = reference_state[:, np.newaxis]

        # Initialize parameters
        self.n_nodes = torch.tensor(n_nodes).to(device)
        if init_weights == 'zero':
            adjacency_weights = np.zeros((self.n_nodes, 1))
        elif init_weights == 'one':
            adjacency_weights = np.ones((self.n_nodes, 1))
        # print(adjacency_weights)
        adjacency_weights = torch.from_numpy(adjacency_weights).to(device)
        self.register_parameter(name='adjacency_weights', param=torch.nn.Parameter(adjacency_weights))

        # Save remaining variables
        self.adjacency_norm = torch.tensor(adjacency_norm).to(device)
        self.time_horizon = torch.tensor(time_horizon).to(device)
        self.control_set = torch.tensor(control_set).to(device)
        self.initial_state = torch.tensor(initial_state).to(device)
        self.target_state = torch.tensor(target_state).to(device)
        self.reference_state = torch.tensor(reference_state).to(device)
        self.rho = torch.tensor(rho).to(device)
        self.trajectory_constraints = torch.tensor(trajectory_constraints).to(device)
        self.I = torch.eye(n_nodes).to(device)
        self.I2 = torch.eye(2 * n_nodes).double().to(device)
        self.O = torch.zeros((n_nodes, n_nodes)).to(device)
        self.IO = torch.concat((self.I, self.O), dim=1).to(device)
        self.eig_weight = eig_weight
        self.reg_weight = reg_weight
        self.reg_type = reg_type


    def eigen_value_loss(self, adjacency_sub):
        # Soft constraint for negative eigenvalues for stability
        # Increase prefactor until max eigenvalue converges to negative number
        eigen_values = torch.linalg.eigvalsh(adjacency_sub)
        eig_val_loss = torch.mean(-torch.div(self.eig_weight, eigen_values))
        
        return eigen_values, eig_val_loss


    def regularization(self, adjacency_sub, type='l2', reg_weight=0.001):
        w = torch.diag(adjacency_sub)
        if type == 'l1':
            return reg_weight * torch.abs(w).sum()
        elif type == 'l2':
            return reg_weight * torch.square(w).sum()


    def forward(self):
        # apply new weights to adjacency
        adjacency_sub = self.adjacency_norm - torch.diag(self.adjacency_weights[:, 0])

        _, eig_val_loss = self.eigen_value_loss(adjacency_sub=adjacency_sub)
        reg = self.regularization(adjacency_sub=adjacency_sub, type=self.reg_type, reg_weight=self.reg_weight)
        loss = reg + eig_val_loss
    
        # define joint state-costate matrix
        M = torch.concat((
            torch.concat((adjacency_sub, torch.matmul(-self.control_set, self.control_set.T) / (2 * self.rho)), dim=1),
            torch.concat((-2 * self.trajectory_constraints, -adjacency_sub.T), dim=1)
        ), dim=0)

        n_transitions = self.initial_state.shape[1]
        for transition_idx in np.arange(n_transitions):
            # define constant vector due to cost deviation from reference state
            c = torch.concat((torch.zeros((self.n_nodes, 1)).to(device),
                              2 * torch.matmul(self.trajectory_constraints, self.reference_state[:, transition_idx:transition_idx + 1])), dim=0)
            c = torch.linalg.solve(M, c)
    
            # compute costate initial condition
            EIp = torch.matrix_exp(M * (self.time_horizon / 2))
            E = torch.matrix_power(EIp, 2)
            r = torch.arange(self.n_nodes)
            E11 = E[r, :][:, r]
            E12 = E[r, :][:, r + self.n_nodes.detach().cpu().numpy()]
            l0 = torch.linalg.solve(E12, (
                        self.target_state[:, transition_idx:transition_idx + 1] - torch.matmul(E11, self.initial_state[:, transition_idx:transition_idx + 1]) - torch.matmul(torch.concat((E11 - self.I, E12), dim=1), c)))
            z0 = torch.concat((self.initial_state[:, transition_idx:transition_idx + 1], l0), dim=0)
    
            # compute intermediate state and error
            EI = self.I2
            EI = torch.matmul(EI, EIp)
            xI = torch.matmul(EI[r, :], z0) + torch.matmul(EI[r, :] - self.IO, c)
            error = torch.linalg.norm(xI - self.reference_state[:, transition_idx:transition_idx + 1])
    
            # Return Loss
            loss += (error / n_transitions)
        return loss
   

def train_nct(adjacency_norm, initial_state, target_state,
              time_horizon=1, control_set='identity',
              reference_state='zero', rho=1, trajectory_constraints='identity',
              init_weights='one',
              n_steps=1000, lr=0.001, eig_weight=0.001, reg_weight=0.001, reg_type='l2',
              early_stopping=True):
    n_nodes = adjacency_norm.shape[0]

    if type(reference_state) == str and reference_state == 'zero':
        reference_state = np.zeros(initial_state.shape)
    elif type(reference_state) == str and reference_state == 'midpoint':
        reference_state = initial_state + ((target_state - initial_state) * 0.5)
    elif type(reference_state) == str and reference_state == 'xf':
        reference_state = target_state

    if type(control_set) == str and control_set == 'identity':
        control_set = np.eye(n_nodes)

    if type(trajectory_constraints) == str and trajectory_constraints == 'identity':
        trajectory_constraints = np.eye(n_nodes)

    nct = NCT(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
              time_horizon=time_horizon, control_set=control_set,
              reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
              init_weights=init_weights,
              eig_weight=eig_weight, reg_weight=reg_weight, reg_type=reg_type)
    opt = torch.optim.Adam(nct.parameters(), lr=lr)
    nct.train()

    loss = np.zeros(n_steps)
    eigen_values = np.zeros(n_steps)
    optimized_weights = np.zeros((n_steps, n_nodes))
    if early_stopping:
        stopping_window = int(n_steps * .20)
    for i in tqdm(np.arange(n_steps)):
        opt.zero_grad()
        nct().backward()
        opt.step()
        loss[i] = nct().detach().cpu().numpy()
        
        optimized_weights[i, :] = nct.adjacency_weights.detach().cpu().numpy().flatten()
        adjacency_sub = adjacency_norm - np.diag(optimized_weights[i, :])
        eigen_values[i] = np.max(np.linalg.eigvalsh(adjacency_sub))
        
        if early_stopping and i > stopping_window:
            loss_var = np.round(np.var(loss[i-stopping_window:i]), 4)
            eig_var = np.round(np.var(eigen_values[i-stopping_window:i]), 4)
            if loss_var == 0 and eig_var == 0:
                print('Hit early stopping criteria (var[loss] = {:.4f}; var[eigen_values] = {:.4f}) at step {:}. Exiting...'.format(loss_var, eig_var, i))
                loss[i+1:] = np.nan
                eigen_values[i+1:] = np.nan
                optimized_weights[i+1:, :] = np.nan
                break
            else:
                pass
        else:
            pass

    return loss, eigen_values, optimized_weights


if __name__ == '__main__':
    start = time.time()
    indir = '/home/lindenmp/research_projects/nct_xr/data'
    outdir = '/home/lindenmp/research_projects/nct_xr/results'
    feature = 'features_schaefer_streamcount_areanorm_log'
    A_file = 'hcp_schaefer400-7_Am-{0}.npy'.format(feature)
    fmri_clusters_file = 'hcp_fmri_clusters_k-7.npy'
    print(A_file, fmri_clusters_file)
    
    # load A matrix
    adjacency = np.load(os.path.join(indir, A_file))

    # load rsfMRI clusters
    fmri_clusters = np.load(os.path.join(outdir, fmri_clusters_file), allow_pickle=True).item()
    centroids = fmri_clusters['centroids']
    [n_states, n_nodes] = centroids.shape

    # nct params
    system = 'continuous'
    c = 1
    time_horizon = 1
    rho = 1
    control_set = np.eye(n_nodes)  # define control set using a uniform full control set
    trajectory_constraints = np.eye(n_nodes)  # define state trajectory constraints

    # normalize adjacency matrix
    adjacency_norm = matrix_normalization(adjacency, system=system, c=c)
    # adjacency_norm = adjacency_norm + np.diag(np.zeros(n_nodes) + 0.5)
    
    # brain states
    noff = 0
    initial_state = np.zeros((centroids.shape[1],np.power(centroids.shape[0]-noff,2)))
    target_state = np.zeros((centroids.shape[1],np.power(centroids.shape[0]-noff,2)))
    transition_idx = 0
    for initial_idx in np.arange(centroids.shape[0] - noff):
        for target_idx in np.arange(centroids.shape[0] - noff):
            initial_state[:, transition_idx] = normalize_state(centroids[initial_idx, :])
            target_state[:, transition_idx] = normalize_state(centroids[target_idx, :])
            transition_idx += 1
    
    reference_state = 'xf'  # reference state
    reference_state_str = reference_state

    # training params
    init_weights = 'one'
    n_steps = 1000  # number of gradient steps
    lr = 0.01  # learning rate for gradient
    eig_weight = 1.0  # regularization strength for eigen value penalty
    reg_weight = 0.0001  # regularization strength for weight penalty (e.g., l2)
    reg_type = 'l2'
    print('training params: init_weights = {0}; n_steps = {1}; lr = {2}; eig_weight = {3}; reg_weight = {4}; reg_type = {5}'.format(init_weights, 
                                                                                                                                    n_steps,
                                                                                                                                    lr, eig_weight, reg_weight, reg_type))
    # run training
    loss, eigen_values, opt_weights = train_nct(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
                                                time_horizon=time_horizon, control_set=control_set,
                                                reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints,
                                                init_weights=init_weights,
                                                n_steps=n_steps, lr=lr, eig_weight=eig_weight, reg_weight=reg_weight, reg_type=reg_type,
                                                early_stopping=True)
    
        # save outputs
    log_args = {
        'loss': loss,
        'eigen_values': eigen_values,
        'opt_weights': opt_weights
    }
    file_str = 'neural_network_c-{0}_T-{1}_rho-{2}_refstate-{3}_weights-{4}_nsteps-{5}_lr-{6}_eigweight-{7}_regweight-{8}_regtype-{9}_{10}'.format(c, time_horizon, rho,
                                                                                                                                                   reference_state_str, init_weights,
                                                                                                                                                   n_steps, lr, eig_weight, reg_weight, reg_type,
                                                                                                                                                   feature)
    print(file_str)
    np.save(os.path.join(outdir, file_str), log_args)

    end = time.time()
    print('...done in {:.2f} seconds.'.format(end - start))
