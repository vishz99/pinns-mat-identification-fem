## To run this file:
## scripts\run_fenics.bat inspect_dataset.py
##
## Loads the generated dataset and prints a summary of its contents.
## Use this to verify the HDF5 file was written correctly before
## moving to PINN training.

import h5py
import numpy as np

DATASET_FILE = "../data/dataset.h5"

with h5py.File(DATASET_FILE, "r") as f:

    # ── Metadata ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    meta = f["metadata"]
    print(f"Total simulations : {meta.attrs['n_samples']}")
    print(f"E range           : [{meta.attrs['E_min']:.2e}, {meta.attrs['E_max']:.2e}] Pa")
    print(f"nu range          : [{meta.attrs['nu_min']:.2f}, {meta.attrs['nu_max']:.2f}]")
    print(f"Train indices     : {len(meta['train_indices'])} cases")
    print(f"Test indices      : {len(meta['test_indices'])} cases")
    print(f"Sensor points     : {meta['sensor_points'].shape}")
    print(f"Grid points       : {meta['grid_points'].shape}")

    # ── Check a few simulations ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE SIMULATIONS")
    print("=" * 60)

    for sim_name in ["sim_0000", "sim_0050", "sim_0199"]:
        grp = f["simulations"][sim_name]
        E   = grp["E"][()]
        nu  = grp["nu"][()]
        u_grid    = grp["u_grid"][()]
        u_sensors = grp["u_sensors"][()]
        split     = grp.attrs["split"]

        print(f"\n{sim_name} [{split}]")
        print(f"  E            : {E:.3e} Pa")
        print(f"  nu           : {nu:.4f}")
        print(f"  u_grid shape : {u_grid.shape}")
        print(f"  u_sensor shape: {u_sensors.shape}")
        print(f"  max u_x      : {u_grid[:, 0].max():.4e} m")
        print(f"  max u_x (analytical estimate): {1e6 * 1.0 / E:.4e} m")

    # ── Parameter coverage ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARAMETER COVERAGE CHECK")
    print("=" * 60)

    all_E  = []
    all_nu = []
    for i in range(200):
        grp = f["simulations"][f"sim_{i:04d}"]
        all_E.append(grp["E"][()])
        all_nu.append(grp["nu"][()])

    all_E  = np.array(all_E)
    all_nu = np.array(all_nu)

    print(f"E  — min: {all_E.min():.3e}  max: {all_E.max():.3e}  mean: {all_E.mean():.3e}")
    print(f"nu — min: {all_nu.min():.4f}  max: {all_nu.max():.4f}  mean: {all_nu.mean():.4f}")