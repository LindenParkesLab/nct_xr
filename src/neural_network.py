import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NNC(nn.Module):
    def __init__(self, A_norm, T, B, x0, xf, xr, rho, S, xmp):
        super().__init__()
        n_nodes = A_norm.shape[0]

        if type(xr) == str and S == 'zero':
            xr = np.zeros((n_nodes, 1))

        if type(S) == str and S == 'identity':
            S = np.eye(n_nodes)

        if type(xmp) == str and xmp == 'midpoint':
            xmp = x0 + ((xf - x0) * 0.5)  # desire intermediate state
        m = xmp.shape[0]
        tmp = np.linspace(0, T, m + 2)[1:-1]

        # Initialize parameters
        self.n_nodes = torch.tensor(n_nodes).to(device)
        self.register_parameter(name='an', param=torch.nn.Parameter(torch.zeros((self.n_nodes, 1)).to(device)))
        # self.register_parameter(name='an', param=torch.nn.Parameter(torch.ones((self.n_nodes, 1)).to(device)))

        # Save remaining variables
        self.A_norm = torch.tensor(A_norm).to(device)
        self.T = torch.tensor(T).to(device)
        self.B = torch.tensor(B).to(device)
        self.x0 = torch.tensor(x0).to(device)
        self.xf = torch.tensor(xf).to(device)
        self.xr = torch.tensor(xr).to(device)
        self.rho = torch.tensor(rho).to(device)
        self.S = torch.tensor(S).to(device)
        self.xmp = torch.tensor(xmp).to(device)
        self.I = torch.eye(n_nodes).to(device)
        self.I2 = torch.eye(2 * n_nodes).double().to(device)
        self.O = torch.zeros((n_nodes, n_nodes)).to(device)
        self.tmp = torch.tensor(tmp).to(device)
        self.m = torch.tensor(m).to(device)
        self.IO = torch.concat((self.I, self.O), dim=1).to(device)

    def forward(self):
        A_sub = self.A_norm - torch.diag(self.an[:, 0])
        # define joint state-costate matrix
        M = torch.concat((
            torch.concat((A_sub, torch.matmul(-self.B, self.B.T) / (2 * self.rho)), dim=1),
            torch.concat((-2 * self.S, -A_sub.T), dim=1)
        ), dim=0)

        # define constant vector due to cost deviation from reference state
        c = torch.concat((torch.zeros((self.n_nodes, 1)).to(device),
                          2 * torch.matmul(self.S, self.xr)), dim=0)
        c = torch.linalg.solve(M, c)

        # compute costate initial condition
        EIp = torch.matrix_exp(M * self.tmp[0])
        E = torch.matrix_power(EIp, self.m + 1)
        r = torch.arange(self.n_nodes)
        E11 = E[r, :][:, r]
        E12 = E[r, :][:, r + self.n_nodes.detach().cpu().numpy()]
        l0 = torch.linalg.solve(E12, (
                    self.xf - torch.matmul(E11, self.x0) - torch.matmul(torch.concat((E11 - self.I, E12), dim=1), c)))
        z0 = torch.concat((self.x0, l0), dim=0)

        # compute intermediate state and error
        L = 0
        EI = self.I2
        for i in torch.arange(self.m):
            EI = torch.matmul(EI, EIp)
            xI = torch.matmul(EI[r, :], z0) + torch.matmul(EI[r, :] - self.IO, c)
            L = L + torch.linalg.norm(xI - self.xmp[i, :, :])
        L = L / self.m

        # Return Loss
        return L

def train_anp(A_norm, T, B, x0, xf, xmp, xr='zero', rho=1, S='identity', n_steps=1000, lr=0.001, verbose=True):
    n_nodes = A_norm.shape[0]

    if type(xr) == str and xr == 'zero':
        xr = np.zeros((n_nodes, 1))

    if type(S) == str and S == 'identity':
        S = np.eye(n_nodes)

    loss = np.zeros(n_steps)
    anp = np.zeros((n_steps, n_nodes))
    nnc = NNC(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf, xr=xr, rho=rho, S=S, xmp=xmp)
    opt = torch.optim.Adam(nnc.parameters(), lr=lr)
    nnc.train()
    if verbose:
        for i in tqdm(np.arange(n_steps)):
            L = nnc()
            opt.zero_grad()
            L.backward()
            opt.step()

            loss[i] = L.detach().cpu().numpy()
            anp[i, :] = nnc.an.detach().cpu().numpy().flatten()
    else:
        for i in np.arange(n_steps):
            L = nnc()
            opt.zero_grad()
            L.backward()
            opt.step()

            loss[i] = L.detach().cpu().numpy()
            anp[i, :] = nnc.an.detach().cpu().numpy().flatten()

    return loss, anp
