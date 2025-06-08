# MLFFwithHessians
# Energy-Force-Hessian (EFH) Training for MLIPs

This repository contains code for training a Machine Learning Interatomic Potential (MLIP) using energy, force, and Hessian matrix data as supervision targets. The model is based on a modified version of ANI and is implemented using PyTorch.

## Overview

This EFH model supports the following training targets:
- **Energies** — total molecular energy.
- **Forces** — first derivatives of energy with respect to atomic positions.
- **Hessians** — second derivatives of energy, providing curvature information about the potential energy surface.

The training workflow is organized in the `EFH-training.ipynb` notebook, which contains:
1. Data loading and preprocessing
2. Model setup and hyperparameters
3. Loss function definitions for energy, force, and Hessian terms
4. Training loop with validation and logging

## Getting Started

### Prerequisites

Ensure the following Python packages are installed:

- `torch`
- `torchani`
- `wandb`
- `numpy`
- `tqdm`

## Cloning the Repository

<pre> git clone https://github.com/mendozacortesgroup/MLFFwithHessians.git
cd MLFFwithHessians </pre>

  ## Modifications Required

To enable the use of Hessian data in training, you must replace two files in the original ANI codebase or MLIP framework:

### Replace the following:

- `utils.py` → place your modified version that supports Hessian loading and batching
- `data/__init__.py` → replace with the provided version that correctly loads and structures Hessian matrix elements

These modifications are essential for:

- Reading Hessian labels from your dataset
- Properly formatting the tensors for automatic differentiation

Make sure these files are replaced **before** running the training notebook.

## Running the notebook

Open and run the notebook
jupyter notebook EFH-training.ipynb

## Notes
- Hessian matrices scale quadratically with system size, so memory and training time considerations are important.
- The model uses automatic differentiation to compute gradients and Hessians efficiently in PyTorch.
- For reproducibility, it is recommended to fix random seeds and document the loss weights used.

## License
