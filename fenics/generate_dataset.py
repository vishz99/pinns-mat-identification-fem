import numpy as np
import h5py
from scipy.stats.qmc import LatinHypercube, scale
from mpi4py import MPI
from forward_solver import run_simulation # import the function

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 200
TRAIN_RATIO = 0.8

# ── Parameter ranges ─────────────────────────────────────────────────────────
E_MIN,  E_MAX  = 50e9,  300e9   # Young's modulus in Pa
NU_MIN, NU_MAX = 0.15,  0.40    # Poisson's ratio

# ── Output file ──────────────────────────────────────────────────────────────
OUTPUT_FILE = "../data/dataset.h5" # .. is going up the folder !!

def sample_parameters(n_samples, seed):
    """
    Generates N combinations of (E, nu) using a Latin hypercube.
    
    Latin hypercube- uniform coverage of the parameter space —
    no clustering, no gaps.
    
    Returns a (N, 2) array where column 0 is E and column 1 is nu.
    """
    sampler = LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    
    scaled_samples = scale(
        unit_samples,
        l_bounds=[E_MIN,  NU_MIN],
        u_bounds=[E_MAX,  NU_MAX]
    )
    
    return scaled_samples


def split_indices(n_samples, train_ratio, seed):
    """
    Randomly shuffles and splits sample indices into train and test sets.
    
    Returns two arrays of indices:
    train_idx : indices of training cases
    test_idx  : indices of test cases
    """
    rng     = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)        #(needed since latin hypercube samples are ordered by their position in parameter space)
    
    n_train     = int(n_samples * train_ratio)
    train_idx   = indices[:n_train]
    test_idx    = indices[n_train:]
    
    return train_idx, test_idx

def generate_dataset(output_file, n_samples, seed, train_ratio):
    """
    Main dataset generation function.
    
    Runs n_samples forward simulations across a Latin hypercube
    of (E, nu) pairs and saves everything to an HDF5 file.
    
    HDF5 structure:
    ├── metadata/            ##### stores everything that is the same across all simulations
    │   ├── sensor_points    (20, 2) — fixed sensor locations
    │   ├── grid_points      (1000, 2) — fixed grid locations  
    │   ├── train_indices    (160,) — which simulations are training cases
    │   └── test_indices     (40,)  — which simulations are test cases
    └── simulations/
        ├── sim_0000/
        │   ├── E            scalar
        │   ├── nu           scalar
        │   ├── u_grid       (1000, 2) — full displacement field
        │   └── u_sensors    (20, 2)   — sparse sensor observations
        ├── sim_0001/
        │   └── ...
        └── sim_0199/
            └── ...
    """
    # ── Sample parameters ────────────────────────────────────────────────────
    params      = sample_parameters(n_samples, seed)
    train_idx, test_idx = split_indices(n_samples, train_ratio, seed)

    # ── Run one simulation first to get fixed point locations ─────────────────
    # sensor_points and grid_points are the same for every simulation
    # (fixed by seed=42 inside run_simulation)
    # We extract them once and store in metadata
    print("Running first simulation to extract fixed point locations...")
    _, _, grid_points, sensor_points = run_simulation(
        E=params[0, 0],
        nu=params[0, 1]
    )

    # ── Open HDF5 file and write metadata ────────────────────────────────────
    with h5py.File(output_file, "w") as f:

        # Metadata group — things that are the same for every simulation
        meta = f.create_group("metadata")
        meta.create_dataset("sensor_points", data=sensor_points)
        meta.create_dataset("grid_points",   data=grid_points)
        meta.create_dataset("train_indices", data=train_idx)
        meta.create_dataset("test_indices",  data=test_idx)
        meta.attrs["n_samples"]   = n_samples
        meta.attrs["train_ratio"] = train_ratio
        meta.attrs["E_min"]       = E_MIN
        meta.attrs["E_max"]       = E_MAX
        meta.attrs["nu_min"]      = NU_MIN
        meta.attrs["nu_max"]      = NU_MAX
        meta.attrs["seed"]        = seed

        # Simulations group — one subgroup per simulation
        sims = f.create_group("simulations")

        # ── Main loop ────────────────────────────────────────────────────────
        for i in range(n_samples):
            E_i  = params[i, 0]
            nu_i = params[i, 1]

            print(f"Simulation {i+1:03d}/{n_samples} — "
                  f"E={E_i:.3e} Pa, nu={nu_i:.4f}")

            u_grid, u_sensors, _, _ = run_simulation(E=E_i, nu=nu_i)

            # Save to HDF5 under simulations/sim_XXXX/
            grp = sims.create_group(f"sim_{i:04d}")
            grp.create_dataset("E",         data=E_i)
            grp.create_dataset("nu",        data=nu_i)
            grp.create_dataset("u_grid",    data=u_grid)
            grp.create_dataset("u_sensors", data=u_sensors)

            split = "train" if i in train_idx else "test"
            # Each simulation subgroup gets a split attribute labelling: "train" or "test". 
            # When dataset is loaded in PyTorch later, 
            # can filter by this attribute rather
            grp.attrs["split"] = split

        print(f"\nDataset complete.")
        print(f"Saved to      : {output_file}")
        print(f"Total cases   : {n_samples}")
        print(f"Training cases: {len(train_idx)}")
        print(f"Test cases    : {len(test_idx)}")


if __name__ == "__main__":
    print("=" * 60)
    print("PINN Inverse Material Identification — Dataset Generation")
    print("=" * 60)
    print(f"Total simulations : {N_SAMPLES}")
    print(f"E range           : [{E_MIN:.2e}, {E_MAX:.2e}] Pa")
    print(f"nu range          : [{NU_MIN:.2f}, {NU_MAX:.2f}]")
    print(f"Train/test split  : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
    print(f"Output file       : {OUTPUT_FILE}")
    print("=" * 60)

    generate_dataset(
        output_file=OUTPUT_FILE,
        n_samples=N_SAMPLES,
        seed=SEED,
        train_ratio=TRAIN_RATIO
    )


## To run this file:
## scripts\run_fenics.bat generate_dataset.py
##
## This runs the full dataset generation pipeline inside the FEniCS Docker container.
## It samples 200 (E, nu) combinations using a Latin hypercube design,
## runs a forward FEM simulation for each combination, and saves all
## displacement fields and sensor observations to data/dataset.h5.
## Run this once before starting PINN training.