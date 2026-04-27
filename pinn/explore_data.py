## Data explorer — inspect the HDF5 dataset generated in Phase 1.
## Prints the structure, contents, and key statistics of the dataset
## so the format is clear before training begins.
##
## Run from project root:
##   python pinn/explore_data.py

import h5py
import numpy as np

DATASET_PATH = "data/dataset.h5"

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def subsection(title):
    print(f"\n  ── {title}")
    print("  " + "-" * 40)

with h5py.File(DATASET_PATH, "r") as f:

    # ── Overall structure ─────────────────────────────────────────────────────
    section("DATASET STRUCTURE")
    print("""
  dataset.h5
  ├── metadata/
  │   ├── sensor_points     fixed 20 sensor coordinates, shape (20, 2)
  │   ├── grid_points       fixed dense grid coordinates, shape (1000, 2)
  │   ├── train_indices     indices of training simulations, shape (160,)
  │   └── test_indices      indices of test simulations, shape (40,)
  │
  └── simulations/
      ├── sim_0000/
      │   ├── E             scalar — Young's modulus used in this simulation
      │   ├── nu            scalar — Poisson's ratio used in this simulation
      │   ├── u_grid        displacement on dense grid, shape (1000, 2)
      │   └── u_sensors     displacement at sensor locations, shape (20, 2)
      ├── sim_0001/ ...
      └── sim_0199/
    """)

    # ── Metadata ──────────────────────────────────────────────────────────────
    section("METADATA")
    meta = f["metadata"]

    subsection("Global attributes")
    for key, val in meta.attrs.items():
        print(f"  {key:<20} : {val}")

    subsection("Sensor points (first 5 of 20)")
    sensor_pts = meta["sensor_points"][()]
    print(f"  Shape : {sensor_pts.shape}  — (n_sensors, 2)")
    print(f"  Columns: x coordinate, y coordinate")
    print(f"\n  {'#':<5} {'x (m)':>10} {'y (m)':>10}")
    print(f"  {'-'*28}")
    for i in range(5):
        print(f"  {i:<5} {sensor_pts[i,0]:>10.4f} {sensor_pts[i,1]:>10.4f}")
    print(f"  ... ({len(sensor_pts)-5} more)")

    subsection("Grid points")
    grid_pts = meta["grid_points"][()]
    print(f"  Shape  : {grid_pts.shape}  — (1000 points, 2 coordinates)")
    print(f"  Layout : 40 points in x × 25 points in y = 1000 total")
    print(f"  x range: [{grid_pts[:,0].min():.3f}, {grid_pts[:,0].max():.3f}] m")
    print(f"  y range: [{grid_pts[:,1].min():.3f}, {grid_pts[:,1].max():.3f}] m")

    subsection("Train / test split")
    train_idx = meta["train_indices"][()]
    test_idx  = meta["test_indices"][()]
    print(f"  Training simulations : {len(train_idx)} cases")
    print(f"  Test simulations     : {len(test_idx)}  cases")
    print(f"  First 5 train indices: {train_idx[:5]}")
    print(f"  First 5 test indices : {test_idx[:5]}")

    # ── Single simulation breakdown ───────────────────────────────────────────
    section("SINGLE SIMULATION BREAKDOWN  (sim_0000)")
    grp = f["simulations"]["sim_0000"]

    subsection("Material parameters")
    E  = float(grp["E"][()])
    nu = float(grp["nu"][()])
    print(f"  E  (Young's modulus) : {E:.4e} Pa  =  {E/1e9:.2f} GPa")
    print(f"  nu (Poisson's ratio) : {nu:.4f}")
    print(f"  Split                : {grp.attrs['split']}")

    subsection("Full displacement field  u_grid  shape (1000, 2)")
    u_grid = grp["u_grid"][()]
    print(f"  Shape   : {u_grid.shape}")
    print(f"  Column 0: u_x — horizontal displacement")
    print(f"  Column 1: u_y — vertical displacement")
    print(f"\n  {'Statistic':<15} {'u_x (m)':>14} {'u_y (m)':>14}")
    print(f"  {'-'*44}")
    print(f"  {'min':<15} {u_grid[:,0].min():>14.4e} {u_grid[:,1].min():>14.4e}")
    print(f"  {'max':<15} {u_grid[:,0].max():>14.4e} {u_grid[:,1].max():>14.4e}")
    print(f"  {'mean':<15} {u_grid[:,0].mean():>14.4e} {u_grid[:,1].mean():>14.4e}")
    print(f"  {'std':<15} {u_grid[:,0].std():>14.4e} {u_grid[:,1].std():>14.4e}")

    subsection("Sensor observations  u_sensors  shape (20, 2)")
    u_sens = grp["u_sensors"][()]
    print(f"  Shape   : {u_sens.shape}")
    print(f"  These are the ONLY values the inverse PINN sees during training.")
    print(f"  The full u_grid field is used only for post-training validation.")
    print(f"\n  {'Sensor':<8} {'x (m)':>8} {'y (m)':>8} {'u_x (m)':>12} {'u_y (m)':>12}")
    print(f"  {'-'*52}")
    for i in range(20):
        print(f"  {i:<8} {sensor_pts[i,0]:>8.4f} {sensor_pts[i,1]:>8.4f} "
              f"{u_sens[i,0]:>12.4e} {u_sens[i,1]:>12.4e}")

    # ── Parameter coverage across all 200 simulations ─────────────────────────
    section("PARAMETER COVERAGE — ALL 200 SIMULATIONS")
    all_E  = np.array([float(f["simulations"][f"sim_{i:04d}"]["E"][()]) for i in range(200)])
    all_nu = np.array([float(f["simulations"][f"sim_{i:04d}"]["nu"][()]) for i in range(200)])

    print(f"\n  {'Statistic':<15} {'E (GPa)':>12} {'nu':>10}")
    print(f"  {'-'*40}")
    print(f"  {'min':<15} {all_E.min()/1e9:>12.3f} {all_nu.min():>10.4f}")
    print(f"  {'max':<15} {all_E.max()/1e9:>12.3f} {all_nu.max():>10.4f}")
    print(f"  {'mean':<15} {all_E.mean()/1e9:>12.3f} {all_nu.mean():>10.4f}")
    print(f"  {'std':<15} {all_E.std()/1e9:>12.3f} {all_nu.std():>10.4f}")

    # ── What the PINN sees vs what it does not ────────────────────────────────
    section("WHAT THE PINN SEES DURING TRAINING")
    print("""
  FORWARD MODE (Phase 2)
  ─────────────────────
  Input to network      : (x, y) coordinates — sampled randomly, no labels
  Physics supervision   : PDE residual at ~5000 collocation points
  Boundary supervision  : Dirichlet (left wall) + Neumann (right, top, bottom)
  Data used from HDF5   : E and nu (fixed known values only)
  Data NOT used         : u_sensors, u_grid

  INVERSE MODE (Phase 3)
  ──────────────────────
  Input to network      : (x, y) coordinates — sampled randomly, no labels
  Physics supervision   : PDE residual at ~5000 collocation points
  Boundary supervision  : Dirichlet (left wall) + Neumann (right, top, bottom)
  Data used from HDF5   : u_sensors (20 observations) to compute data-fit loss
  Data NOT used         : u_grid (reserved for post-training validation only)
  Recovered parameters  : E and nu (learnable scalars updated by optimiser)
    """)

print("\nExploration complete.")