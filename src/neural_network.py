import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NNC(nn.Module):
    def __init__(self, adjacency_norm, initial_state, target_state,
                 time_horizon, control_set,
                 reference_state, rho, trajectory_constraints):
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
            reference_state = reference_state[np.newaxis, :, np.newaxis]
        elif reference_state.ndim == 2:
            reference_state = reference_state[:, :, np.newaxis]
        # print(initial_state.shape, target_state.shape, reference_state.shape)

        n_ref_states = reference_state.shape[0]
        tmp = np.linspace(0, time_horizon, n_ref_states + 2)[1:-1]
        reference_state_mean = reference_state.mean(axis=0)
        # print(reference_state_mean.shape)

        # Initialize parameters
        self.n_nodes = torch.tensor(n_nodes).to(device)
        self.register_parameter(name='adjacency_weights', param=torch.nn.Parameter(torch.zeros((self.n_nodes, 1)).to(device)))

        # Save remaining variables
        self.adjacency_norm = torch.tensor(adjacency_norm).to(device)
        self.time_horizon = torch.tensor(time_horizon).to(device)
        self.control_set = torch.tensor(control_set).to(device)
        self.initial_state = torch.tensor(initial_state).to(device)
        self.target_state = torch.tensor(target_state).to(device)
        self.reference_state = torch.tensor(reference_state).to(device)
        self.n_ref_states = torch.tensor(n_ref_states).to(device)
        self.reference_state_mean = torch.tensor(reference_state_mean).to(device)
        self.rho = torch.tensor(rho).to(device)
        self.trajectory_constraints = torch.tensor(trajectory_constraints).to(device)
        self.I = torch.eye(n_nodes).to(device)
        self.I2 = torch.eye(2 * n_nodes).double().to(device)
        self.O = torch.zeros((n_nodes, n_nodes)).to(device)
        self.tmp = torch.tensor(tmp).to(device)
        self.IO = torch.concat((self.I, self.O), dim=1).to(device)

    def forward(self):
        # apply new weights to adjacency
        adjacency_sub = self.adjacency_norm - torch.diag(self.adjacency_weights[:, 0])
        # normalize adjacency

        # define joint state-costate matrix
        M = torch.concat((
            torch.concat((adjacency_sub, torch.matmul(-self.control_set, self.control_set.T) / (2 * self.rho)), dim=1),
            torch.concat((-2 * self.trajectory_constraints, -adjacency_sub.T), dim=1)
        ), dim=0)

        # define constant vector due to cost deviation from reference state
        c = torch.concat((torch.zeros((self.n_nodes, 1)).to(device),
                          2 * torch.matmul(self.trajectory_constraints, self.reference_state_mean)), dim=0)
        c = torch.linalg.solve(M, c)

        # compute costate initial condition
        EIp = torch.matrix_exp(M * self.tmp[0])
        E = torch.matrix_power(EIp, self.n_ref_states + 1)
        r = torch.arange(self.n_nodes)
        E11 = E[r, :][:, r]
        E12 = E[r, :][:, r + self.n_nodes.detach().cpu().numpy()]
        l0 = torch.linalg.solve(E12, (
                    self.target_state - torch.matmul(E11, self.initial_state) - torch.matmul(torch.concat((E11 - self.I, E12), dim=1), c)))
        z0 = torch.concat((self.initial_state, l0), dim=0)

        # compute intermediate state and error
        loss = 0
        EI = self.I2
        for i in torch.arange(self.n_ref_states):
            EI = torch.matmul(EI, EIp)
            xI = torch.matmul(EI[r, :], z0) + torch.matmul(EI[r, :] - self.IO, c)
            loss = loss + torch.linalg.norm(xI - self.reference_state[i, :, :])
        loss = loss / self.n_ref_states

        # Return Loss
        return loss

def train_anp(adjacency_norm, initial_state, target_state,
              time_horizon=1, control_set='identity',
              reference_state='zero', rho=1, trajectory_constraints='identity',
              n_steps=1000, lr=0.001):
    n_nodes = adjacency_norm.shape[0]

    if type(reference_state) == str and reference_state == 'zero':
        reference_state = np.zeros((1, n_nodes, 1))
    elif type(reference_state) == str and reference_state == 'midpoint':
        reference_state = initial_state + ((target_state - initial_state) * 0.5)
    elif type(reference_state) == str and reference_state == 'target_state':
        reference_state = target_state
    if reference_state.ndim == 1:
        reference_state = reference_state[np.newaxis, :, np.newaxis]

    if type(control_set) == str and control_set == 'identity':
        control_set = np.eye(n_nodes)

    if type(trajectory_constraints) == str and trajectory_constraints == 'identity':
        trajectory_constraints = np.eye(n_nodes)

    loss = np.zeros(n_steps)
    optimized_weights = np.zeros((n_steps, n_nodes))
    nnc = NNC(adjacency_norm=adjacency_norm, initial_state=initial_state, target_state=target_state,
              time_horizon=time_horizon, control_set=control_set,
              reference_state=reference_state, rho=rho, trajectory_constraints=trajectory_constraints)
    opt = torch.optim.Adam(nnc.parameters(), lr=lr)
    nnc.train()
    for i in np.arange(n_steps):
        opt.zero_grad()
        nnc().backward()
        opt.step()
        nnc.adjacency_weights.data.clamp_(min=-0.9, max=2.0)

        loss[i] = nnc().detach().cpu().numpy()
        optimized_weights[i, :] = nnc.adjacency_weights.detach().cpu().numpy().flatten()

    return loss, optimized_weights
