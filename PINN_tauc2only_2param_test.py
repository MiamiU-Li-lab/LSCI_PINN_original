#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:45:04 2025
@author: shyli
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os

# --------- Model Definition ---------
class InverseModel(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    def forward(self, y):
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        # Raw predictions for [rho0, tauc2]
        raw = self.fc3(y)
        # Constrain rho0 to (0,1)
        rho0 = torch.sigmoid(raw[:, 0:1])
        # Constrain tauc2 to be positive
        tauc2 = nn.functional.softplus(raw[:, 1:2])
        return torch.cat([rho0, tauc2], dim=1)

# --------- Physical Model Function ---------
def physical_model(x, T):
    rho0 = x[:, 0:1]
    tauc2 = x[:, 1:2]
    beta0 = x[:, 2:3]
    eps = 1e-12
    x2 = T / (tauc2 + eps)
    
    term1 = (rho0 ** 2) * (torch.exp(-2 * x2) - 1 + 2 * x2) / (2 * x2 ** 2)
    term2 = 4 * rho0 * (1 - rho0) * (torch.exp(-x2) - 1 + x2) / (x2 ** 2)
    term3 = (1 - rho0) ** 2

    result = beta0.sqrt() * (term1 + term2 + term3).sqrt()
    return result

# --------- R² Computation Function ---------
def compute_r2_map(Y_true, Y_pred, H, W):
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=1)
    ss_tot = ((Y_true - Y_true.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1 - ss_res / ss_tot
    return r2.reshape(H, W)

# --------- Main Script ---------
test_dir  = '../08_22_BL18'
mat_files = sorted(glob.glob(os.path.join(test_dir, 'LSCI*slow*.mat')))
output_dir   = 'results_slow_dynamics_BL14_model'
os.makedirs(output_dir, exist_ok=True)

# Load T from the first file
with h5py.File(mat_files[0], 'r') as f:
    T = np.array(f['P']['Texp'])[7:]
T = torch.tensor(T, dtype=torch.float32).unsqueeze(0).squeeze(-1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InverseModel().to(device)
model.load_state_dict(torch.load('PINN_state_dict_slowdynamics.pth', map_location=device))
model.eval()


for file in mat_files:
    fname = os.path.splitext(os.path.basename(file))[0]
    print(f'Processing {fname}...')

    with h5py.File(file, 'r') as f:
        data = np.array(f['mK'])  # [C, W, H]

    data = np.expand_dims(data, axis=0)  # [1, H, W, C]
    data = data[:, :, :, 7:]  # drop timepoints before 1000ms
    H, W = data.shape[1], data.shape[2]
    Y_input = data.reshape(-1, data.shape[-1])
    Y_tensor = torch.tensor(Y_input, dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        pred_params = model(Y_tensor)  # [N, 2]
        pred_params = pred_params.cpu()
        pred_rho0 = pred_params[:, 0].reshape(H, W).numpy()
        pred_tauc2 = pred_params[:, 1].reshape(H, W).numpy()
        beta0 = (Y_tensor[:, 0] ** 2).cpu().numpy().reshape(H, W)

        # Predict Y via physics model
        beta0_tensor = (Y_tensor[:, 0:1] ** 2)
        x_pred = torch.cat([pred_params.to(device), beta0_tensor.to(device)], dim=1)
        T_exp = T.expand(x_pred.shape[0], -1).to(device)
        Y_pred = physical_model(x_pred, T_exp).cpu().numpy()
        Y_true = Y_tensor.cpu().numpy()

        # Compute R² map
        r2_map = compute_r2_map(Y_true, Y_pred, H, W)
        avg_r2 = np.mean(r2_map)
        print(f'Average R² for {fname}: {avg_r2:.4f}')
    # --- Plot all 4 maps ---
    maps = [pred_rho0, pred_tauc2, beta0, r2_map]
    titles = ['Predicted rho0', 'Predicted tauc2', 'True beta_0', 'R^2 Map']
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        d = maps[i]
        vmin, vmax = np.percentile(d, [1, 99])
        im = axs[i].imshow(np.clip(d, vmin, vmax), cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir + f'/{fname}_allmaps.png', dpi=300)
    plt.close()
    