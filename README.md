# PINN-Based Inverse Solver for Material Parameter Recovery in 2D Linear Elasticity

A Physics-Informed Neural Network that recovers elastic material parameters: Young's modulus E and Poisson's ratio ν, from sparse displacement observations by embedding the 2D linear elasticity PDE directly into the training loss.

---

## Introduction

This project implements a PINN-based inverse solver for 2D linear elasticity in PyTorch. A feedforward neural network represents the displacement field as a continuous function of spatial coordinates; automatic differentiation computes strain and stress exactly, and the equilibrium PDE is enforced at collocation points throughout the domain. Material parameters E and ν are recovered simultaneously with the displacement field by treating them as learnable scalars optimised against sparse FEM-generated displacement observations.

Ground truth data is generated using FEniCS (dolfinx), a Python finite element library running inside Docker. A parameter sweep of 200 simulations across Young's modulus E ∈ [50, 300] GPa and Poisson's ratio ν ∈ [0.15, 0.40], sampled using a Latin hypercube design, produces the training and test dataset stored in HDF5 format.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project foundation, theory, and setup | Complete |
| 1 | FEniCS forward solver and dataset generation | Complete |
| 2 | Forward PINN — validate PDE loss with known E and ν | In Progress |
| 3 | Inverse PINN — recover E and ν from sparse observations | Pending |
| 4 | Repository cleanup, results notebook, and final documentation | Pending |

---

## Project Structure
```
pinn-material-identification/
├── data/                        HDF5 dataset (not tracked — see data/README.md to regenerate)
├── fenics/
│   ├── forward_solver.py        FEM solver for a single (E, ν) pair
│   ├── generate_dataset.py      Parameter sweep loop — generates data/dataset.h5
│   └── inspect_dataset.py       Validates the generated HDF5 dataset
├── pinn/                        PyTorch model, loss terms, training loop, and evaluation
├── configs/                     Hyperparameter configuration files
├── notebooks/                   Results walkthrough notebook
├── scripts/
│   └── run_fenics.bat           Windows wrapper for running FEniCS scripts via Docker
└── outputs/                     Checkpoints, plots, and logs (not tracked)
```
---

---

## Dataset

| Property | Value |
|----------|-------|
| Total simulations | 200 |
| E range | [50, 300] GPa |
| ν range | [0.15, 0.40] |
| Sampling method | Latin hypercube |
| Training cases | 160 (80%) |
| Test cases | 40 (20%) |
| Sensor locations per simulation | 20 (fixed across all simulations) |
| Grid points per simulation | 1000 (40 × 25 regular grid) |
| File format | HDF5 via h5py |

To regenerate the dataset see `data/README.md`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Primary language |
| PyTorch | Neural network, training loop, automatic differentiation |
| FEniCSx (dolfinx) 0.10.0 | Finite element forward solver for ground truth data generation |
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