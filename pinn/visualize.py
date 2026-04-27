## Visualisation and result interpretation tools.
## Run after training to inspect loss curves, displacement fields,
## and parameter recovery results.
##
## Usage:
##   python pinn/visualize.py --history outputs/checkpoints/RUN_ID/history.json
##                            --dataset data/dataset.h5
##                            --sim_index 0
## python pinn/visualize.py --history outputs\checkpoints\20260424_160659\history.json --dataset data\dataset.h5 --sim_index 0
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import torch
from model import PINN


def plot_loss_curves(history, save_dir):
    """
    Plots all loss terms on a single figure with a log scale y-axis.
    Each term is plotted separately so convergence behaviour is visible
    for each physical requirement independently.

    A healthy training run shows:
        - All terms decreasing monotonically or near-monotonically
        - No single term dominating at the expense of others
        - L_total flattening as convergence is reached
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    terms = [
        ("L_total", "Total Loss",     axes[0, 0], "#1F4E79"),
        ("L_pde",   "PDE Residual",   axes[0, 1], "#C55A11"),
        ("L_dir",   "Dirichlet BC",   axes[1, 0], "#2E75B6"),
        ("L_neu",   "Neumann BC",     axes[1, 1], "#1D6A6A"),
    ]

    for key, label, ax, color in terms:
        if key in history and len(history[key]) > 0:
            ax.semilogy(history[key], color=color, linewidth=1.2)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#F8F9FA")

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_parameter_recovery(history, E_true, nu_true, save_dir):
    """
    Plots the evolution of recovered E and nu over training epochs.
    Only meaningful in inverse mode — skipped if no recovery history exists.

    Shows how quickly and accurately the optimiser converges to the
    true material parameter values from the initial guess.
    Horizontal dashed lines mark the true values for reference.
    """
    if not history.get("E_recovered") or not history.get("nu_recovered"):
        print("No parameter recovery history found — forward mode training.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Material Parameter Recovery", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["E_recovered"]) + 1)

    # Young's modulus
    ax1.plot(epochs, [e / 1e9 for e in history["E_recovered"]],
             color="#C55A11", linewidth=1.5, label="Recovered E")
    ax1.axhline(E_true / 1e9, color="#1F4E79", linewidth=1.5,
                linestyle="--", label=f"True E = {E_true/1e9:.1f} GPa")
    ax1.set_title("Young's Modulus E", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("E (GPa)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#F8F9FA")

    # Poisson's ratio
    ax2.plot(epochs, history["nu_recovered"],
             color="#2E75B6", linewidth=1.5, label="Recovered ν")
    ax2.axhline(nu_true, color="#1F4E79", linewidth=1.5,
                linestyle="--", label=f"True ν = {nu_true:.4f}")
    ax2.set_title("Poisson's Ratio ν", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ν")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#F8F9FA")

    plt.tight_layout()
    path = os.path.join(save_dir, "parameter_recovery.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_displacement_comparison(model_path, dataset_path, sim_index, save_dir):
    """
    Compares the PINN predicted displacement field against the FEniCS
    ground truth on the dense 40x25 evaluation grid.

    Plots four panels:
        u_x FEniCS    — ground truth horizontal displacement
        u_x PINN      — network predicted horizontal displacement
        u_y FEniCS    — ground truth vertical displacement
        u_y PINN      — network predicted vertical displacement

    The two fields should look nearly identical if the PINN has converged.
    Large visible differences indicate insufficient training or loss
    weight issues that need tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config     = checkpoint["config"]

    model = PINN(n_hidden=config["n_hidden"], n_neurons=config["n_neurons"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load FEniCS ground truth
    sim_key = f"sim_{sim_index:04d}"
    with h5py.File(dataset_path, "r") as f:
        grid_xy = f["metadata"]["grid_points"][()]
        u_grid  = f["simulations"][sim_key]["u_grid"][()]
        E_true  = float(f["simulations"][sim_key]["E"][()])
        nu_true = float(f["simulations"][sim_key]["nu"][()])

    # PINN prediction on grid
    xy_tensor = torch.tensor(grid_xy, dtype=torch.float32).to(device)
    with torch.no_grad():
        uv_pred = model(xy_tensor).cpu().numpy()

    # Reshape to 2D grid for plotting (25 rows x 40 cols)
    x = grid_xy[:, 0].reshape(25, 40)
    y = grid_xy[:, 1].reshape(25, 40)
    ux_fem  = u_grid[:, 0].reshape(25, 40)
    uy_fem  = u_grid[:, 1].reshape(25, 40)
    ux_pinn = uv_pred[:, 0].reshape(25, 40)
    uy_pinn = uv_pred[:, 1].reshape(25, 40)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Displacement Field Comparison — sim_{sim_index:04d}\n"
        f"E = {E_true/1e9:.2f} GPa,  ν = {nu_true:.4f}",
        fontsize=13, fontweight="bold"
    )

    panels = [
        (axes[0, 0], ux_fem,  "u_x — FEniCS (ground truth)", "m"),
        (axes[0, 1], ux_pinn, "u_x — PINN prediction",       "m"),
        (axes[1, 0], uy_fem,  "u_y — FEniCS (ground truth)", "m"),
        (axes[1, 1], uy_pinn, "u_y — PINN prediction",       "m"),
    ]

    for ax, data, title, unit in panels:
        cf = ax.contourf(x, y, data, levels=50, cmap="RdBu_r")
        plt.colorbar(cf, ax=ax, label=unit)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(save_dir, "displacement_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def print_summary_table(history, E_true, nu_true, checkpoint_path):
    """
    Prints a clean summary table of training results to the terminal.
    Gives a quick overview of final loss values and parameter recovery
    accuracy without needing to open any plots.
    """
    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device("cpu"),
                            weights_only=False)

    E_recovered  = checkpoint.get("E",  None)
    nu_recovered = checkpoint.get("nu", None)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"\n{'Loss Term':<20} {'Final Value':>15}")
    print("-" * 36)
    for key in ["L_total", "L_pde", "L_dir", "L_neu", "L_data"]:
        if key in history and len(history[key]) > 0:
            print(f"{key:<20} {history[key][-1]:>15.4e}")

    if E_recovered is not None and E_true is not None:
        print(f"\n{'Parameter':<15} {'True':>12} {'Recovered':>12} {'Error':>10}")
        print("-" * 52)
        err_E  = abs(E_recovered - E_true)  / E_true  * 100
        err_nu = abs(nu_recovered - nu_true) / nu_true * 100
        print(f"{'E (GPa)':<15} {E_true/1e9:>12.4f} {E_recovered/1e9:>12.4f} {err_E:>9.2f}%")
        print(f"{'nu':<15} {nu_true:>12.4f} {nu_recovered:>12.4f} {err_nu:>9.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",   type=str, required=True,
                        help="Path to history.json from training run")
    parser.add_argument("--dataset",   type=str, default="data/dataset.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--sim_index", type=int, default=0,
                        help="Simulation index to compare against")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt checkpoint")
    args = parser.parse_args()

    # Derive checkpoint path from history path if not explicitly provided
    run_dir = os.path.dirname(args.history)
    checkpoint_path = args.checkpoint or os.path.join(run_dir, "best_model.pt")

    # Load history
    with open(args.history, "r") as f:
        history = json.load(f)

    # Load true values from dataset
    with h5py.File(args.dataset, "r") as f:
        sim_key = f"sim_{args.sim_index:04d}"
        E_true  = float(f["simulations"][sim_key]["E"][()])
        nu_true = float(f["simulations"][sim_key]["nu"][()])

    # Save plots to same directory as the history file
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    save_dir = os.path.join(run_dir, "plots")

    # Generate all outputs
    plot_loss_curves(history, save_dir)
    plot_parameter_recovery(history, E_true, nu_true, save_dir)
    plot_displacement_comparison(checkpoint_path, args.dataset,
                                 args.sim_index, save_dir)
    print_summary_table(history, E_true, nu_true, checkpoint_path)