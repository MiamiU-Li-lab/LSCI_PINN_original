#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 04:59:01 2025
@author: shyli
"""
import os, glob, h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
# --- Physical Model ---
def physical_model(x, T):
    rho0 = x[:, 0:1]
    tauc2 = x[:, 1:2]
    beta0 = x[:, 2:3]
    x2 = T / (tauc2 + 1e-10)
    eps = 1e-10
    term1 = (rho0 ** 2) * (torch.exp(-2 * x2) - 1 + 2 * x2) / (2 * x2 ** 2 + eps)
    term2 = 4 * rho0 * (1 - rho0) * (torch.exp(-x2) - 1 + x2) / (x2 ** 2 + eps)
    term3 = (1 - rho0) ** 2
    return beta0.sqrt() * (term1 + term2 + term3).sqrt()

def physics_loss(y_hat, y_true, T):
    beta0 = (y_true[:, 0] ** 2).unsqueeze(1)
    x_combined = torch.cat([y_hat, beta0], dim=1)
    y_pred = physical_model(x_combined, T)
    return nn.functional.mse_loss(y_pred, y_true)

# --- Load Data ---
train_data_dir = 'data/BL14'
mat_files = sorted(glob.glob(os.path.join(train_data_dir, 'LSCI*slow*.mat')))

X_list = []
with h5py.File(mat_files[0], 'r') as f:
    T = np.array(f['P']['Texp'])[7:]  # cut short-T
T = torch.tensor(T, dtype=torch.float32).unsqueeze(0)

for file in mat_files:
    with h5py.File(file, 'r') as f:
        mK = np.array(f['mK'])
        mK = np.expand_dims(mK, axis=0)  # [1, H, W, C]
        X_list.append(mK)

train_img_indices = [0, 1, 2, 3, 4, 6, 7, 8]
test_img_indices = [5, 9]

X = np.concatenate(X_list, axis=0)[:, :, :, 7:]  # [N, H, W, C]
H, W = X.shape[1:3]
Y_train = X[train_img_indices].reshape(-1, X.shape[-1])
Y_test = X[test_img_indices].reshape(-1, X.shape[-1])
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# --- Dataset ---
train_loader = DataLoader(TensorDataset(Y_train, Y_train), batch_size=32768, shuffle=True)
test_loader = DataLoader(TensorDataset(Y_test, Y_test), batch_size=32768, shuffle=False)

# --- Inverse Model ---
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


# %% train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InverseModel().to(device)
optimizer = torch.optim.Rprop(model.parameters(), lr=1e-2)

train_losses, test_losses = [], []
epochs = 200

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_train_loss = 0.0
    for Y_batch, _ in train_loader:
        Y_batch = Y_batch.to(device)
        y_hat = model(Y_batch)
        T_batch = T.squeeze(-1).expand(Y_batch.size(0), -1).to(device)
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
            T_batch = T.squeeze(-1).expand(Y_batch.size(0), -1).to(device)
            loss = physics_loss(y_hat, Y_batch, T_batch)
            total_test_loss += loss.item()
    test_losses.append(total_test_loss / len(test_loader))
    time_spent = time.time() - start_time
    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.6f} | Test Loss: {test_losses[-1]:.6f} | Time: {time_spent:.2f}s")

# %% validation and visualization
start_time = time.time()
model.eval()
outputs = []
num_images = 2
# Inference
with torch.no_grad():
    for Y_batch, _ in test_loader:
        Y_batch = Y_batch.to(device)
        y_hat = model(Y_batch)
        outputs.append(y_hat.cpu())

outputs = torch.cat(outputs, dim=0)  # [num_pixels, 2]
outputs_image = outputs.reshape(num_images, H, W, 2)

# True beta0
with torch.no_grad():
    true_beta0_all = (Y_test[:, 0]**2).reshape(num_images, H, W).numpy()

# Make results directory
os.makedirs('results', exist_ok=True)

titles = ['Predicted rho0', 'Predicted tauc2', 'beta0 (K[:,0]^2)']

for idx in range(num_images):
    pred_rho0 = outputs_image[idx, :, :, 0].numpy()
    pred_tauc2 = outputs_image[idx, :, :, 1].numpy()
    beta0 = true_beta0_all[idx]

    all_data = [pred_rho0, pred_tauc2, beta0]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        data = all_data[i]
        vmin, vmax = np.percentile(data, [1, 99])
        clipped_data = np.clip(data, vmin, vmax)
        im = axs[i].imshow(clipped_data, cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'prediction_img{idx:02d}.png', dpi=300)
    plt.close()

test_duration = time.time() - start_time
print(f"Test Time: {test_duration:.2f} sec")
print(f"Saved {num_images} prediction result images")
torch.save(model.state_dict(), 'PINN_state_dict_slowdynamics.pth')
