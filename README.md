# PINN-Based Inverse Solver for Material Parameter Recovery in 2D Linear Elasticity

A Physics-Informed Neural Network that recovers elastic material parameters — Young's modulus E and Poisson's ratio ν — from sparse displacement observations by embedding the 2D linear elasticity PDE directly into the training loss.

---

## Introduction

This project implements a PINN-based inverse solver for 2D linear elasticity in PyTorch. A feedforward neural network represents the displacement field as a continuous function of spatial coordinates; automatic differentiation computes strain and stress exactly, and the equilibrium PDE is enforced at collocation points throughout the domain. Material parameters E and ν are recovered simultaneously with the displacement field by treating them as learnable scalars optimised against sparse FEM-generated displacement observations.

Ground truth data is generated using FEniCS (dolfinx), a Python finite element library running inside Docker.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project foundation, theory, and setup | Complete |
| 1 | FEniCS forward solver and dataset generation | In Progress |
| 2 | Forward PINN — validate PDE loss with known E and ν | Pending |
| 3 | Inverse PINN — recover E and ν from sparse observations | Pending |
| 4 | Repository cleanup, results notebook, and final documentation | Pending |

---

## Project Structure
```
pinn-material-identification/
├── data/               HDF5 dataset (not tracked — see data/README.md to regenerate)
├── fenics/             FEniCS forward solver and dataset generation scripts
├── pinn/               PyTorch model, loss terms, training loop, and evaluation
├── configs/            Hyperparameter configuration files
├── notebooks/          Results walkthrough notebook
├── scripts/            Shell wrappers for Docker and training runs
└── outputs/            Checkpoints, plots, and logs (not tracked)
```
---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Primary language |
| PyTorch | Neural network, training loop, automatic differentiation |
| FEniCSx (dolfinx) | Finite element forward solver for ground truth data generation |
| Docker | Containerised FEniCS environment |
| h5py | HDF5 dataset storage and loading |
| NumPy | Array operations |
| SciPy | Latin hypercube parameter sampling |
| Matplotlib | Plotting and visualisation |

---

## Requirements
```
pip install -r requirements.txt
```
FEniCS runs inside Docker and is not installed via pip. See `data/README.md` for setup instructions.