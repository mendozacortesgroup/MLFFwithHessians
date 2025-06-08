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


