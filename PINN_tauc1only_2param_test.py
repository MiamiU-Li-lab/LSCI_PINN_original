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
import time

# --------- Model Definition ---------
class InverseModel(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    def forward(self, y):
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        raw = self.fc3(y)
        rho1  = torch.sigmoid(raw[:, 0:1])
        tauc1 = nn.functional.softplus(raw[:, 1:2])
        return torch.cat([rho1, tauc1], dim=1)

# --- Physical Model ---
def physical_model(x, T):
    rho1 = x[:, 0:1]
    tauc2 = x[:, 1:2]
    beta0 = torch.tensor(0.72)
    x1 = T / tauc2
    sqrt_x1 = torch.sqrt(x1)
    x1_sq = x1 ** 2
    eps = 1e-10

    A = torch.exp(-2 * sqrt_x1) * (4 * x1 + 6 * sqrt_x1 + 3) - 3 + 2 * x1
    term1 = (rho1 ** 2) * A / (2 * x1_sq + eps)

    B = torch.exp(-sqrt_x1) * (2 * x1 + 6 * sqrt_x1 + 6) - 6 + x1
    term2 = 8 * rho1 * (1 - rho1) * B / (x1_sq + eps)

    term3 = (1 - rho1) ** 2

    result = torch.sqrt(beta0) * torch.sqrt(term1 + term2 + term3)
    return result  

# --------- R² Computation Function ---------
def compute_r2_map(Y_true, Y_pred, H, W):
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=1)
    ss_tot = ((Y_true - Y_true.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1 - ss_res / (ss_tot)
    return r2.reshape(H, W)

# --------- Main Script ---------
test_data_dir = '08_22_BL18'
mat_files = sorted([
    f for f in glob.glob(os.path.join(test_data_dir, 'LSCI_*_WFfast_*.mat'))
    if f.count("_") == 7   # exactly 3 underscores total
])
# Load T from the first file
with h5py.File(mat_files[0], 'r') as f:
    T = np.array(f['P']['Texp'])
T = torch.tensor(T, dtype=torch.float32).unsqueeze(0).squeeze(-1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InverseModel().to(device)
model.load_state_dict(torch.load('PINN_state_dict_fastdynamics_BL14.pth', map_location=device))
model.eval()

# Create output directory
output_dir = 'results_fast_dynamics_BL14_model'
os.makedirs(output_dir, exist_ok=True)
start_time = time.time()
for file in mat_files:
    fname = os.path.splitext(os.path.basename(file))[0]
    print(f'Processing {fname}...')

    with h5py.File(file, 'r') as f:
        data = np.array(f['mK'])  # [C, W, H]

    data = np.expand_dims(data, axis=0)  # [1, H, W, C]
    H, W = data.shape[1], data.shape[2]
    Y_input = data.reshape(-1, data.shape[-1])
    Y_tensor = torch.tensor(Y_input, dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        pred_params = model(Y_tensor)  # [N, 2]
        pred_params = pred_params.cpu()
        pred_rho0 = pred_params[:, 0].reshape(H, W).numpy()
        pred_tauc1 = pred_params[:, 1].reshape(H, W).numpy()

        x_pred = pred_params.to(device)
        T_exp = T.expand(x_pred.shape[0], -1).to(device)
        Y_pred = physical_model(x_pred, T_exp).cpu().numpy()
        Y_true = Y_tensor.cpu().numpy()

        r2_map = compute_r2_map(Y_true, Y_pred, H, W)
        avg_r2 = np.mean(r2_map)
        print(f'Average R\u00b2 for {fname}: {avg_r2:.4f}')

    # --- Plot all 4 maps ---
    beta0 = np.full((H, W), 0.72)
    maps = [pred_rho0, pred_tauc1, beta0, r2_map]
    titles = ['Predicted \u03c1\u2080', 'Predicted \u03c4_c1', '\u03b2\u2080', 'R\u00b2 Map']
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        d = maps[i]
        vmin, vmax = np.percentile(d, [1, 99])
        im = axs[i].imshow(np.clip(d, vmin, vmax), cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    fig.suptitle(f'Model Output for {fname}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir + f'/{fname}_allmaps.png', dpi=300)
    plt.close()

time_spent = time.time() - start_time
print(f'Total inference time: {time_spent:.2f} sec')
print('All done.')
