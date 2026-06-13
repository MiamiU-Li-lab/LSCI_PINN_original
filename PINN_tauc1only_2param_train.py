#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 04:59:01 2025
@author: shyli
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob 
import os
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

# --- Physics-guided Loss ---
def physics_loss(y_hat, y_true, T):
    y_pred = physical_model(y_hat, T)
    return nn.functional.mse_loss(y_pred, y_true)

# --- Load Data ---
train_data_dir = 'BL14'
mat_files = sorted(glob.glob(os.path.join(train_data_dir, 'LSCI*fast*.mat')))

X_list = []
with h5py.File(mat_files[0], 'r') as f:
    T = np.array(f['P']['Texp'])  # cut short-T
T = torch.tensor(T, dtype=torch.float32).unsqueeze(0)

for file in mat_files:
    with h5py.File(file, 'r') as f:
        mK = np.array(f['mK'])
        mK = np.expand_dims(mK, axis=0)  # [1, H, W, C]
        X_list.append(mK)

train_img_indices = [0, 1, 2, 3, 4, 6, 7, 8]
test_img_indices = [5, 9]

X = np.concatenate(X_list, axis=0) # [N, H, W, C]
H, W = X.shape[1:3]
Y_train = X[train_img_indices].reshape(-1, X.shape[-1])
Y_test = X[test_img_indices].reshape(-1, X.shape[-1])
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# --- DataLoader ---
BATCH_SIZE = 32768
train_loader = DataLoader(TensorDataset(Y_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(Y_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- Inverse Model ---
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
        
# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InverseModel().to(device)
optimizer = torch.optim.Rprop(model.parameters(), lr=1e-2)

train_losses = []
test_losses = []
epochs = 200

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    for Y_batch, _ in train_loader:
        Y_batch = Y_batch.to(device)
        y_hat = model(Y_batch)
        T_batch = T.repeat(Y_batch.size(0), 1, 1).squeeze(-1).to(device)
        loss = physics_loss(y_hat, Y_batch, T_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for Y_batch, _ in test_loader:
            Y_batch = Y_batch.to(device)
            y_hat = model(Y_batch)
            T_batch = T.repeat(Y_batch.size(0), 1, 1).squeeze(-1).to(device)
            loss = physics_loss(y_hat, Y_batch, T_batch)
            total_test_loss += loss.item()
    test_losses.append(total_test_loss / len(test_loader))

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Test Loss: {test_losses[-1]:.6f}")
# %% validation & visualization
start_time = time.time()
model.eval()
outputs = []
num_images = 2

with torch.no_grad():
    for Y_batch, _ in test_loader:
        Y_batch = Y_batch.to(device)
        y_hat = model(Y_batch)
        outputs.append(y_hat.cpu())

outputs = torch.cat(outputs, dim=0).reshape(num_images, H, W, 2)

os.makedirs('results', exist_ok=True)
titles = ['Predicted rho0', 'Predicted tauc1']

for idx in range(num_images):
    pred_rho0 = outputs[idx, :, :, 0].numpy()
    pred_tauc2 = outputs[idx, :, :, 1].numpy()
    all_data = [pred_rho0, pred_tauc2]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(2):
        data = all_data[i]
        vmin, vmax = np.percentile(data, [1, 99])
        im = axs[i].imshow(np.clip(data, vmin, vmax), cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'prediction_img{idx:02d}.png', dpi=300)
    plt.show()
    plt.close()

print(f"Test Time: {time.time() - start_time:.2f} sec")
print(f"Saved {num_images} prediction result images")

torch.save(model.state_dict(), 'PINN_state_dict_fastdynamics_BL14.pth')
