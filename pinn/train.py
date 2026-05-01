## Physics-Informed Neural Network — Training Loop
##
## Handles both forward mode (Phase 2) and inverse mode (Phase 3).
## Mode is determined by whether E and nu are passed as fixed floats
## or as nn.Parameter objects.
##
## To run:
##   python pinn/train.py --config configs/base.yaml

import sys
import os

from pinn import visualize
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
# ^ These are relative imports that assume the working directory is pinn/. 
# But I am running from the project root, so Python will not find those files.
import torch
import numpy as np
import yaml
import argparse
import os
import json
from datetime import datetime
from model import PINN
from loss import total_loss


# ═══════════════════════════════════════════════════════════════════════════
# COLLOCATION AND BOUNDARY POINT SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def sample_collocation_points(n_points, device):
    """
    Samples random interior collocation points uniformly inside the domain.

    Domain: x ∈ [0, 1], y ∈ [0, 0.5]

    These points carry no labels — the PDE residual = 0 is the supervision
    signal. Points are resampled every epoch to prevent the network from
    overfitting to a fixed set of locations.

    Returns torch.Tensor shape (n_points, 2) with requires_grad=True.
    """
    xy = torch.zeros(n_points, 2, device=device)
    xy[:, 0] = torch.rand(n_points, device=device) * 1.0          # x ∈ [0, 1]
    xy[:, 1] = torch.rand(n_points, device=device) * 0.5          # y ∈ [0, 0.5]
    xy.requires_grad_(True)
    return xy


def sample_boundary_points(n_points, device):
    """
    Samples points on each boundary edge.

    Returns four tensors, all with requires_grad=True:
        xy_left   : x=0,   y uniform in [0, 0.5]   — Dirichlet (clamped)
        xy_right  : x=1,   y uniform in [0, 0.5]   — Neumann (traction)
        xy_top    : y=0.5, x uniform in [0, 1]     — Neumann (free)
        xy_bottom : y=0,   x uniform in [0, 1]     — Neumann (free)
    """
    def make_boundary(fixed_dim, fixed_val, free_range, n, dev):
        pts = torch.zeros(n, 2, device=dev)
        pts[:, fixed_dim] = fixed_val
        pts[:, 1 - fixed_dim] = torch.rand(n, device=dev) * free_range
        pts.requires_grad_(True)
        return pts

    xy_left   = make_boundary(0, 0.0, 0.5, n_points, device)
    xy_right  = make_boundary(0, 1.0, 0.5, n_points, device)
    xy_top    = make_boundary(1, 0.5, 1.0, n_points, device)
    xy_bottom = make_boundary(1, 0.0, 1.0, n_points, device)

    return xy_left, xy_right, xy_top, xy_bottom


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(config, E_true=None, nu_true=None,
          xy_sensors=None, u_obs=None,
          inverse_mode=False):
    """
    Main training loop. Handles both forward and inverse modes.

    Forward mode  (Phase 2): E and nu are fixed floats taken from config.
                             No sensor data used.
                             Goal: verify PDE loss drives network to correct
                             displacement field.

    Inverse mode  (Phase 3): E and nu are nn.Parameter objects initialised
                             to a guess. Sensor observations are passed in.
                             Goal: recover true E and nu from sparse data.

    Parameters
    ----------
    config       : dict — hyperparameters loaded from YAML config file
    E_true       : float — true Young's modulus (used for logging error only)
    nu_true      : float — true Poisson's ratio (used for logging error only)
    xy_sensors   : torch.Tensor, shape (M, 2) — sensor coordinates
    u_obs        : torch.Tensor, shape (M, 2) — observed displacements
    inverse_mode : bool — if True, E and nu become learnable parameters

    Returns
    -------
    model      : trained PINN
    history    : dict of loss and parameter history for plotting
    """
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = PINN(
        n_hidden=config["n_hidden"],
        n_neurons=config["n_neurons"]
    ).to(device)

    # ── Material parameters ──────────────────────────────────────────────────
    ########################################################################### important
    if inverse_mode:
        # E and nu are unknowns — initialise to guess and make learnable
        # Log-parametrisation of E ensures E stays positive during optimisation
        log_E = torch.log(torch.tensor(config["E_init"],
                          dtype=torch.float32, device=device))
        log_E = torch.nn.Parameter(log_E)
        nu    = torch.nn.Parameter(torch.tensor(config["nu_init"],
                                   dtype=torch.float32, device=device))
        params = list(model.parameters()) + [log_E, nu]
        print(f"Inverse mode — initial E: {config['E_init']:.3e}, "
              f"initial nu: {config['nu_init']:.4f}")
    else:
        # E and nu are fixed known values — not part of the optimisation
        log_E = None
        E     = torch.tensor(config["E_true"], dtype=torch.float32, device=device)
        nu    = torch.tensor(config["nu_true"], dtype=torch.float32, device=device)
        params = model.parameters()
        print(f"Forward mode — E: {config['E_true']:.3e}, "
              f"nu: {config['nu_true']:.4f}")
    ###########################################################################
    # ── Optimiser ────────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(params, lr=config["learning_rate"])

    # ── Learning rate scheduler ──────────────────────────────────────────────
    # Reduces learning rate by factor 0.5 if total loss does not improve
    # for patience epochs. Helps fine convergence in later training stages.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"]
    )

    # ── Loss weights ─────────────────────────────────────────────────────────
    lam_pde  = config["lambda_pde"]
    lam_dir  = config["lambda_dir"]
    lam_neu  = config["lambda_neu"]
    lam_data = config["lambda_data"]

    # ── Move sensor data to device ───────────────────────────────────────────
    if xy_sensors is not None:
        xy_sensors = xy_sensors.to(device)
        u_obs      = u_obs.to(device)

    # ── History tracking ─────────────────────────────────────────────────────
    history = {
        "L_total": [], "L_pde": [], "L_dir": [],
        "L_neu": [], "L_data": [],
        "E_recovered": [], "nu_recovered": []
    }

    # ── Output directory ─────────────────────────────────────────────────────
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join("outputs", "checkpoints", run_id)
    os.makedirs(out_dir, exist_ok=True)
    best_loss = float("inf")

    # ── Training loop ────────────────────────────────────────────────────────
    n_epochs        = config["n_epochs"]
    n_collocation   = config["n_collocation"]
    n_boundary      = config["n_boundary"]
    log_every       = config["log_every"]

    for epoch in range(1, n_epochs + 1):

        optimiser.zero_grad()

        # Resample collocation and boundary points every epoch
        # Resampling prevents network from memorising a fixed point set
        # and encourages uniform PDE enforcement across the domain
        xy_col                              = sample_collocation_points(n_collocation, device)
        xy_left, xy_right, xy_top, xy_bot  = sample_boundary_points(n_boundary, device)

        # Recover E from log-parametrisation in inverse mode
        E = torch.exp(log_E) if inverse_mode else E

        # Compute total loss and individual components
        loss, parts = total_loss(
            model=model,
            xy_collocation=xy_col,
            xy_left=xy_left,
            xy_right=xy_right,
            xy_top=xy_top,
            xy_bottom=xy_bot,
            E=E,
            nu=nu,
            traction=config["traction"],
            xy_sensors=xy_sensors if inverse_mode else None,
            u_obs=u_obs      if inverse_mode else None,
            lambda_pde=lam_pde,
            lambda_dir=lam_dir,
            lambda_neu=lam_neu,
            lambda_data=lam_data
        )
        """
        Added new!
        """
        # Soft constraint: penalise nu outside physical range [0.15, 0.40]
        # nu_penalty = (torch.relu(nu - 0.40) + torch.relu(0.15 - nu)) ** 2
        # loss = loss + 100.0 * nu_penalty

        # Backward pass and optimiser step
        loss.backward()
        optimiser.step()
        scheduler.step(loss)

        # Hard clamp nu to physical range after each optimiser step
        with torch.no_grad():
            nu.clamp_(0.15, 0.40)

        # ── Logging ──────────────────────────────────────────────────────────
        for key, val in parts.items():
            if key in history:
                history[key].append(val)

        if inverse_mode:
            E_curr  = torch.exp(log_E).item()
            nu_curr = nu.item()
            history["E_recovered"].append(E_curr)
            history["nu_recovered"].append(nu_curr)

        if epoch % log_every == 0 or epoch == 1:
            log_str = (f"Epoch {epoch:05d}/{n_epochs} | "
                       f"L_total: {parts['L_total']:.4e} | "
                       f"L_pde: {parts['L_pde']:.4e} | "
                       f"L_dir: {parts['L_dir']:.4e} | "
                       f"L_neu: {parts['L_neu']:.4e}")
            if inverse_mode:
                log_str += (f" | E: {E_curr:.4e} | nu: {nu_curr:.4f}")
                if E_true is not None:
                    err_E  = abs(E_curr - E_true) / E_true * 100
                    err_nu = abs(nu_curr - nu_true) / nu_true * 100
                    log_str += (f" | E_err: {err_E:.2f}% | "
                                f"nu_err: {err_nu:.2f}%")
            print(log_str)

        # ── Checkpoint — save best model ─────────────────────────────────────
        if parts["L_total"] < best_loss:
            best_loss = parts["L_total"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "loss": best_loss,
                "E": torch.exp(log_E).item() if inverse_mode else E.item(),
                "nu": nu.item(),
                "config": config
            }, os.path.join(out_dir, "best_model.pt"))

    # ── Save training history ─────────────────────────────────────────────────
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4e}")
    print(f"Checkpoint saved to: {out_dir}")

    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

# Parses command line arguments, loads the YAML config, and reads the
# specified simulation from the HDF5 dataset generated in Phase 1.
# E_true and nu_true are always taken from the dataset — not from the config —
# to ensure the forward PINN trains against the exact FEniCS solution.
# Sensor observations are passed to the training loop only in inverse mode.
# Toggle between forward and inverse mode with the --inverse flag:
#   Forward : python pinn/train.py --config configs/base.yaml
#   Inverse : python pinn/train.py --config configs/base.yaml --inverse

"""
The training loop is controlled by inverse_mode. 
When False — E and ν are fixed tensors, sensor data is not passed to total_loss, 
and only model parameters are in the optimiser. 
When True — E is a log-parametrised nn.Parameter, ν is an nn.Parameter, both are added to the 
optimiser's parameter list alongside the model weights, and sensor data activates the data-fit loss term.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the PINN in forward or inverse mode."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Run in inverse mode — E and nu become learnable parameters"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)      ###### yaml file is opened and refered to here.

    # ── Load dataset ──────────────────────────────────────────────────────────
    # Dataset is needed in both modes:
    #   Forward mode : loads E_true and nu_true from the chosen simulation
    #                  to use as fixed known parameters and for validation
    #   Inverse mode : additionally loads sensor observations u_obs which
    #                  the network must match through the data-fit loss term
    import h5py

    dataset_path = config["dataset_path"]
    sim_index    = config["sim_index"]
    sim_key      = f"sim_{sim_index:04d}"

    print(f"Loading simulation {sim_key} from {dataset_path}")

    with h5py.File(dataset_path, "r") as f:
        E_true  = float(f["simulations"][sim_key]["E"][()])
        nu_true = float(f["simulations"][sim_key]["nu"][()])

        # Sensor locations and observed displacements
        sensor_xy = f["metadata"]["sensor_points"][()]   # shape (20, 2)
        u_sensors = f["simulations"][sim_key]["u_sensors"][()]  # shape (20, 2)

        # Full displacement field for post-training validation
        grid_xy   = f["metadata"]["grid_points"][()]     # shape (1000, 2)
        u_grid    = f["simulations"][sim_key]["u_grid"][()] # shape (1000, 2)
        sim_split = f["simulations"][sim_key].attrs["split"]

    print(f"Simulation {sim_key} — E: {E_true:.3e} Pa, nu: {nu_true:.4f}")
    print(f"Split: {sim_split}")

    # Convert sensor data to tensors for training
    xy_sensors_tensor = torch.tensor(sensor_xy, dtype=torch.float32)
    u_obs_tensor      = torch.tensor(u_sensors, dtype=torch.float32)

    # Override config E_true and nu_true with actual dataset values
    # This ensures forward mode uses the correct physics for this simulation
    config["E_true"]  = E_true
    config["nu_true"] = nu_true

    # ── Run training ──────────────────────────────────────────────────────────
    model, history = train(
        config=config,
        E_true=E_true,
        nu_true=nu_true,
        xy_sensors=xy_sensors_tensor if args.inverse else None,
        u_obs=u_obs_tensor           if args.inverse else None,
        inverse_mode=args.inverse
    )

    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════════
# USAGE GUIDE
# ═══════════════════════════════════════════════════════════════════════════
#
# FORWARD MODE (Phase 2)
# ──────────────────────
# E and nu are fixed known values loaded from the dataset.
# The network learns the displacement field using physics alone —
# no sensor observations are used.
# Purpose: verify the PDE loss implementation is correct before
# attempting parameter recovery. If the PINN cannot reproduce the
# FEniCS field with known material parameters, something is wrong
# in the loss function — fix it here before moving to inverse mode.
#
# Input    : (x, y) spatial coordinates — sampled randomly inside the domain
#            E and nu — fixed known values loaded from the dataset (not learned)
#
# Output   : Trained displacement field u(x, y) = (u_x, u_y)
#            Loss curves for L_pde, L_dir, L_neu, L_total
#            best_model.pt checkpoint and history.json saved to outputs/
#
# Recovered: Nothing — E and nu are fixed. Success means the predicted
#            displacement field visually matches the FEniCS ground truth.
#
#   python pinn/train.py --config configs/base.yaml
#
#
# INVERSE MODE (Phase 3)
# ──────────────────────
# E and nu are unknown — initialised to a guess and made learnable.
# The network simultaneously learns the displacement field AND recovers
# the material parameters from 20 sparse sensor observations.
# The data-fit loss term activates and drives E and nu to their true values.
# Only run this after forward mode has been validated successfully.
# Input    : (x, y) spatial coordinates — sampled randomly inside the domain
#            u_sensors — 20 observed displacements loaded from the dataset
#            E_init, nu_init — initial guesses defined in configs/base.yaml
#
# Output   : Trained displacement field u(x, y) = (u_x, u_y)
#            Recovered E and nu values printed every log_every epochs
#            Loss curves for L_pde, L_dir, L_neu, L_data, L_total
#            best_model.pt checkpoint and history.json saved to outputs/
#
# Recovered: E (Young's modulus in Pa) and nu (Poisson's ratio)
#            Recovery error printed as percentage against true values
#            Parameter convergence history saved in history.json
#
#   python pinn/train.py --config configs/base.yaml --inverse
#
#
# VISUALISING RESULTS
# ───────────────────
# After training, a run folder is created under outputs/checkpoints/RUN_ID/
# containing best_model.pt and history.json. Pass these to visualize.py:
#
#   python pinn/visualize.py --history outputs/checkpoints/RUN_ID/history.json
#                            --dataset data/dataset.h5
#                            --sim_index 0
#
# ═══════════════════════════════════════════════════════════════════════════
# Example:
# python pinn/visualize.py --history outputs\checkpoints\20260501_083123\history.json --dataset data\dataset.h5 --sim_index 1