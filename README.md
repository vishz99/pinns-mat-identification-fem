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
| 2 | Forward PINN — validate PDE loss with known E and ν | Complete |
| 3 | Inverse PINN — recover E and ν from sparse observations | Complete |
| 4 | Repository cleanup, results notebook, and final documentation | Pending |

---

## Project Structure
 
Numbers in parentheses indicate the chronological execution order across phases.
 
```
pinn-material-identification/
│
├── data/
│   ├── .gitkeep                        ← Preserves folder in Git
│   └── README.md                       ← Instructions to regenerate dataset
│                                         dataset.h5 is gitignored — regenerate via scripts\run_fenics.bat
│
├── fenics/
│   ├── forward_solver.py               ← (1.0) FEM solver for a single (E, ν) pair
│   ├── generate_dataset.py             ← (1.1) Parameter sweep — generates data/dataset.h5
│   └── inspect_dataset.py              ← (1.2) Validates the generated HDF5 dataset
│
├── pinn/
│   ├── explore_data.py                 ← (2.0) Inspect dataset structure before training
│   ├── model.py                        ← (2.1) PINN architecture — FFNN with tanh activations
│   ├── loss.py                         ← (2.2) All four loss terms: PDE, Dirichlet, Neumann, data-fit
│   ├── train.py                        ← (2.3) Training loop — forward and inverse mode
│   ├── visualize.py                    ← (2.4) Loss curves, displacement field comparison, results table
│   └── evaluate.py                     ← (2.5) Batch evaluation across multiple simulations
│
├── configs/
│   └── base.yaml                       ← (2.3) All hyperparameters — passed to train.py at runtime
│
├── scripts/
│   └── run_fenics.bat                  ← (1.0) Windows wrapper for running FEniCS scripts via Docker
│
├── notebooks/
│   └── results_walkthrough.ipynb       ← (4.0) End-to-end results walkthrough (Phase 4)
│
├── outputs/                            ← Gitignored — populated during training
│   ├── checkpoints/RUN_ID/
│   │   ├── best_model.pt               ← Best model weights saved during training
│   │   └── history.json                ← Loss and parameter history per epoch
│   └── plots/                          ← Training curves and displacement field comparisons
│
├── .gitignore
├── README.md
├── requirements.txt                    ← pip install -r requirements.txt
└── requirements_fenics.txt             ← FEniCS runs in Docker — reference only
```
---
 
## Execution Order
 
### Phase 1 — Data Generation (runs inside Docker)
 
```bash
# Run FEniCS forward solver — validates single simulation
scripts\run_fenics.bat forward_solver.py
 
# Generate full dataset of 200 simulations
scripts\run_fenics.bat generate_dataset.py
 
# Validate dataset contents
scripts\run_fenics.bat inspect_dataset.py
```
 
### Phase 2 — Forward PINN (runs locally on GPU)
 
```bash
# Inspect dataset structure before training
python pinn/explore_data.py
 
# Train forward PINN — fixed E and ν, physics only
python pinn/train.py --config configs/base.yaml
 
# Visualise results
python pinn/visualize.py --history outputs\checkpoints\RUN_ID\history.json \
                         --dataset data\dataset.h5 --sim_index 0
```
 
### Phase 3 — Inverse PINN (runs locally on GPU)
 
```bash
# Train inverse PINN — recover E and ν from sparse observations
python pinn/train.py --config configs/base.yaml --inverse
 
# Visualise results
python pinn/visualize.py --history outputs\checkpoints\RUN_ID\history.json \
                         --dataset data\dataset.h5 --sim_index 1
```
 
---
 
## Key Results
 
| Simulation | E true | E recovered | E error | ν true | ν recovered | ν error |
|------------|--------|-------------|---------|--------|-------------|---------|
| sim_0001 | 178.9 GPa | 187.6 GPa | 4.82% | 0.2404 | 0.3034 | 26.22% |
| *(more coming)* | | | | | | |
 
E is recovered to within 5% across tested simulations. ν recovery is harder — the uniaxial displacement field is primarily sensitive to E; ν only manifests through the smaller lateral contraction u_y. With randomly placed sensors, the gradient signal for ν is weak. This is a physically meaningful limitation, not an implementation error — see the discussion in the Phase 3 documentation.
 
---
 
## Governing Equations
 
Equilibrium PDE enforced at 5000 collocation points:
 
```
∂σ_xx/∂x + ∂σ_xy/∂y = 0
∂σ_xy/∂x + ∂σ_yy/∂y = 0
```
 
Hooke's law in plane stress (E and ν appear here — recovered in inverse mode):
 
```
σ_xx = E/(1-ν²) · (ε_xx + ν·ε_yy)
σ_yy = E/(1-ν²) · (ε_yy + ν·ε_xx)
σ_xy = E/(1+ν)  · ε_xy
```
 
Total training loss:
 
```
L = λ_pde · L_pde + λ_dir · L_dir + λ_neu · L_neu + λ_data · L_data
```
 
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
| Sensor locations | 20 fixed points per simulation |
| Grid points | 1000 (40 × 25 regular grid) |
| File format | HDF5 via h5py |
 
---
 
## Tech Stack
 
| Tool | Purpose |
|------|---------|
| Python 3.x | Primary language |
| PyTorch 2.x + CUDA 12.4 | PINN training, automatic differentiation |
| FEniCSx (dolfinx) 0.10.0 | FEM forward solver for ground truth data |
| Docker | Containerised FEniCS environment |
| h5py | HDF5 dataset storage and loading |
| NumPy | Array operations |
| SciPy | Latin hypercube parameter sampling |
| Matplotlib | Loss curves, field plots, results visualisation |
 
---
 
## Requirements
 
```bash
pip install -r requirements.txt
```
 
FEniCS runs inside Docker — not installed via pip. See `data/README.md` for Docker setup instructions.
 
---
 
## Known Limitations and Future Work
 
- **ν sensitivity**: Random sensor placement in uniaxial tension gives weak gradient signal for ν. Placing sensors along top and bottom edges where lateral contraction is maximum would significantly improve ν recovery.
- **Sensor count**: Increasing from 20 to 50 sensors improves coverage and reduces the probability of poor placement.
- **Biaxial loading**: Applying loads in both x and y directions makes E and ν equally observable from any sensor placement.
- **Strain observations**: Using strain gauge readings (ε directly) rather than displacement gives a stronger and more direct gradient signal for ν recovery.
---
 
## Tags
 
physics-informed-neural-network pinn inverse-problem solid-mechanics linear-elasticity
pytorch fenics scientific-machine-learning automatic-differentiation finite-element-method
surrogate-model material-identification computational-mechanics deep-learning python