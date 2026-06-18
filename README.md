# LSCI-PINN: Physics-Informed Neural Network for Laser Speckle Contrast Imaging

[![GitHub](https://img.shields.io/badge/GitHub-MU--Li--lab%2FLSCI__PINN__original-blue?logo=github)](https://github.com/MU-Li-lab/LSCI_PINN_original)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Source repository:** https://github.com/MU-Li-lab/LSCI_PINN_original

---

## Overview

This repository provides the **original baseline** Physics-Informed Neural Network (PINN) scripts for estimating cerebral blood flow parameters from wide-field **Laser Speckle Contrast Imaging (LSCI)** data.

Two independent models invert speckle contrast decay curves **K(T)** measured across multiple exposure times into physically interpretable flow parameters:

| Model | Target parameter | Script |
|---|---|---|
| Fast-dynamics | τ<sub>c1</sub>, ρ<sub>1</sub>  | `PINN_tauc1only_2param_train.py` |
| Slow-dynamics | τ<sub>c2</sub>, ρ<sub>0</sub> | `PINN_tauc2only_2param_train.py` |

The forward physics models are embedded in the loss function, enabling unsupervised parameter recovery directly from raw speckle contrast measurements — no ground-truth parameter labels are required.

---

## Physics Background

### Fast-Dynamics Model 


$$K(T) = \sqrt{\beta_0} \cdot \sqrt{\rho_1^2 \frac{A}{2x_1^2} + 8\rho_1(1-\rho_1)\frac{B}{x_1^2} + (1-\rho_1)^2}$$

where $x_1 = T/\tau_{c1}$, $\sqrt{x_1} = \sqrt{T/\tau_{c1}}$, and:

$$A = e^{-2\sqrt{x_1}}\left(4x_1 + 6\sqrt{x_1} + 3\right) - 3 + 2x_1$$

$$B = e^{-\sqrt{x_1}}\left(2x_1 + 6\sqrt{x_1} + 6\right) - 6 + x_1$$

β<sub>0</sub> is fixed at 0.72. Estimated parameters: **{ρ<sub>1</sub>, τ<sub>c1</sub>}**.

### Slow-Dynamics Model 

$$K(T) = \sqrt{\beta_0} \cdot \sqrt{\rho_0^2 \frac{e^{-2x_2}-1+2x_2}{2x_2^2} + 4\rho_0(1-\rho_0)\frac{e^{-x_2}-1+x_2}{x_2^2} + (1-\rho_0)^2}$$

where $x_2 = T/\tau_{c2}$. Estimated parameters: **{ρ<sub>0</sub>, τ<sub>c2</sub>}**, with β<sub>0</sub> derived from data as K(T<sub>0</sub>)<sup>2</sup>.

---

## Repository Structure

```
LSCI_PINN_original/
├── PINN_tauc1only_2param_train.py         # Fast-dynamics PINN — training
├── PINN_tauc1only_2param_test.py    # Fast-dynamics PINN — evaluation
├── PINN_tauc2only_2param_train.py   # Slow-dynamics PINN — training
├── PINN_tauc2only_2param_test.py    # Slow-dynamics PINN — evaluation
├── environment.yml                  # Conda environment (pinn_lsci)
├── data/
│   ├── BL14/                        # Training data (mouse #14, baseline session)
│   └── BL13/                        # Evaluation data (mouse #13)
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MU-Li-lab/LSCI_PINN_original.git
cd LSCI_PINN_original
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate pinn_lsci
```

> **GPU note:** The `environment.yml` pins `pytorch-cuda=11.8`. Adjust this to match your local CUDA driver version (e.g. `12.1`) if needed. CPU-only execution is supported automatically when no CUDA device is detected.

---

## Data Preparation

multi-exposure, LSCI speckle contrast maps are stored as **MATLAB v7.3 (HDF5) `.mat`** files. 

### Expected directory layout

```
LSCI_PINN_original/
└── data/
    ├── BL14/                        # Training data (mouse #14, baseline session)
    │   ├── LSCI*fast*.mat
    │   └── LSCI*slow*.mat
    └── BL13/                        # Evaluation data (mouse #13)
        ├── LSCI_*_WFfast_*.mat
        └── LSCI*slow*.mat
```

Each `.mat` file contains:

| Key | Shape | Description |
|---|---|---|
| `mK` | `[C × H × W]` | Speckle contrast maps across C exposure times |
| `P/Texp` | `[C]` | Exposure times in milliseconds |

---

## Training

### Fast-Dynamics Model

```bash
python PINN_tauc1only_2param_train.py
```

- Reads `data/BL14/LSCI*fast*.mat`  
- 20 exposure timepoints (full T range)  
- Outputs: `PINN_state_dict_fastdynamics.pth`

### Slow-Dynamics Model

```bash
python PINN_tauc2only_2param_train.py
```

- Reads `data/BL14/LSCI*slow*.mat`  
- 28 exposure timepoints (first 7 dropped: T < ~1000 ms)  
- Outputs: `PINN_state_dict_slowdynamics.pth`

---

## Evaluation

### Fast-Dynamics Model

```bash
python PINN_tauc1only_2param_test.py
```

- Loads `PINN_state_dict_fastdynamics.pth`  
- Tests on `data/BL13/LSCI*fast*.mat`  
- Saves to `results_fast_dynamics_BL14_model/`

### Slow-Dynamics Model

```bash
python PINN_tauc2only_2param_test.py
```

- Loads `PINN_state_dict_slowdynamics.pth`  
- Tests on `data/BL13/LSCI*slow*.mat`  
- Saves to `results_slow_dynamics_BL14_model/`

---

## Outputs

Each evaluation run produces the following per input file:

| File | Contents |
|---|---|
| `*_allmaps.png` | 4-panel image: predicted ρ map, predicted τ<sub>c</sub> map, true β<sub>0</sub> map, R² map |

---

## Model Architecture

Both models are **pointwise fully connected networks** — each pixel's contrast decay curve is processed independently.

### Fast-Dynamics Network

| Layer | Input → Output |
|---|---|
| Linear + ReLU | 20 → 128 |
| Linear + ReLU | 128 → 64 |
| Linear | 64 → 2 raw |
| sigmoid | raw[:, 0] → ρ<sub>1</sub> ∈ (0, 1) |
| softplus | raw[:, 1] → τ<sub>c1</sub> > 0 |

### Slow-Dynamics Network

| Layer | Input → Output |
|---|---|
| Linear + ReLU | 28 → 128 |
| Linear + ReLU | 128 → 64 |
| Linear | 64 → 2 raw |
| sigmoid | raw[:, 0] → ρ<sub>0</sub> ∈ (0, 1) |
| softplus | raw[:, 1] → τ<sub>c2</sub> > 0 |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this code or data in your research, please cite our work. BibTeX will be updated upon publication.
@article{Li2026pinnlsci,
journal = {bioRxiv},
publisher = {Cold Spring Harbor Laboratory},
title = {{Physics-Informed Neural Network for Mapping Vascular and Tissue Dynamics Using Laser Speckle Contrast Imaging}},
url = {https://www.biorxiv.org/content/early/2026/02/02/2026.02.01.702939},
year = {2026}
}

---

## Contact

**MU Li Lab** 
GitHub: [https://github.com/MU-Li-lab](https://github.com/MU-Li-lab)
