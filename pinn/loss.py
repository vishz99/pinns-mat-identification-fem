## Physics-Informed Neural Network — Loss Functions
##
## Implements all four loss terms:
##   L_pde  : PDE residual (equilibrium equations at collocation points)
##   L_dir  : Dirichlet BC (zero displacement on clamped left wall)
##   L_neu  : Neumann BC (applied traction on right edge, free top/bottom)
##   L_data : Data-fit term (sensor observations, inverse mode only)
##
## To run training:
##   python pinn/train.py --config configs/base.yaml

import torch

# ── Normalisation constants ───────────────────────────────────────────────────
# All loss terms are normalised to order 1 before weighting.
# Without normalisation, stress values (~1e11) and displacement values (~1e-6)
# differ by 17 orders of magnitude, making loss balancing impossible.
U_SCALE = 1e-6    # displacement scale in metres
S_SCALE = 1e6     # stress scale in Pascals (= traction magnitude)
E_SCALE = 1e11    # stiffness scale in Pascals


# ═══════════════════════════════════════════════════════════════════════════
# KINEMATICS AND CONSTITUTIVE LAW
# ═══════════════════════════════════════════════════════════════════════════

def compute_strain(u_x, u_y, xy):
    """
    Computes the plane stress strain tensor components from the displacement
    field using automatic differentiation.

    Strain is the symmetric part of the displacement gradient:
        ε_xx = ∂u_x/∂x
        ε_yy = ∂u_y/∂y
        ε_xy = 0.5 * (∂u_x/∂y + ∂u_y/∂x)

    Parameters
    ----------
    u_x : torch.Tensor, shape (N, 1) — x-displacement component
    u_y : torch.Tensor, shape (N, 1) — y-displacement component
    xy  : torch.Tensor, shape (N, 2) — input coordinates, requires_grad=True

    Returns
    -------
    eps_xx, eps_yy, eps_xy : torch.Tensor, each shape (N, 1)
    """
    # Gradient of u_x with respect to both x and y
    grad_ux = torch.autograd.grad(
        outputs=u_x,
        inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,    # needed for second-order derivatives later
        retain_graph=True
    )[0]                      # shape (N, 2) — columns are ∂u_x/∂x and ∂u_x/∂y

    # Gradient of u_y with respect to both x and y
    grad_uy = torch.autograd.grad(
        outputs=u_y,
        inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,    # Critical step: tells PyTorch to build a graph of the gradient computation itself, so it can be differentiated again.
        retain_graph=True     # allows multiple backward passes on the same graph, which is necessary since we will compute second derivatives for the PDE residual loss.
    )[0]                      # shape (N, 2) — columns are ∂u_y/∂x and ∂u_y/∂y

    eps_xx = grad_ux[:, 0:1]                          # ∂u_x/∂x
    eps_yy = grad_uy[:, 1:2]                          # ∂u_y/∂y
    eps_xy = 0.5 * (grad_ux[:, 1:2] + grad_uy[:, 0:1])  # ½(∂u_x/∂y + ∂u_y/∂x)

    return eps_xx, eps_yy, eps_xy


def compute_stress(eps_xx, eps_yy, eps_xy, E, nu):
    """
    Computes normalised plane stress components from strain using Hooke's law.

    Plane stress constitutive law:
        σ_xx = E/(1-ν²) * (ε_xx + ν*ε_yy)
        σ_yy = E/(1-ν²) * (ε_yy + ν*ε_xx)
        σ_xy = E/(1+ν)  * ε_xy

    Normalisation:
        Network outputs normalised displacements (divided by U_SCALE).
        Autograd gives strain in units of U_SCALE/metre.
        Physical stress = E * physical_strain.
        Normalised stress = physical_stress / S_SCALE.
        Combined scale = E_SCALE * U_SCALE / S_SCALE = 1e11*1e-6/1e6 = 0.1

    Returned stresses are order 0.1 to 0.3 — suitable for loss computation.

    Parameters
    ----------
    eps_xx, eps_yy, eps_xy : strain components (normalised)
    E  : float or torch.Tensor — Young's modulus in Pa (unnormalised)
    nu : float or torch.Tensor — Poisson's ratio (dimensionless, no scaling)

    Returns
    -------
    sig_xx, sig_yy, sig_xy : normalised stress components (order 1)
    """
    E_norm = E / E_SCALE
    scale  = E_SCALE * U_SCALE / S_SCALE   # = 0.1

    factor = scale * E_norm / (1.0 - nu ** 2)

    sig_xx = factor * (eps_xx + nu * eps_yy)
    sig_yy = factor * (eps_yy + nu * eps_xx)
    sig_xy = scale * E_norm / (1.0 + nu) * eps_xy

    return sig_xx, sig_yy, sig_xy

# ═══════════════════════════════════════════════════════════════════════════
# LOSS TERMS
# ═══════════════════════════════════════════════════════════════════════════

# Requirement 1: The material must be in equilibrium everywhere inside Ω
# PDE loss:  the most important function in the project — enforces the equilibrium equations at collocation points.
"""
xy → model → (u_x, u_y)
          → compute_strain → (ε_xx, ε_yy, ε_xy)     [1st derivatives of u]
          → compute_stress → (σ_xx, σ_yy, σ_xy)     [Hooke's law]
          → autograd again → (∂σ/∂x, ∂σ/∂y)         [2nd derivatives of u]
          → r1, r2                                    [PDE residuals]

"""
def loss_pde(model, xy_collocation, E, nu):
    """
    PDE residual loss — enforces static equilibrium at collocation points.

    The equilibrium equations (no body forces):
        r1 = ∂σ_xx/∂x + ∂σ_xy/∂y = 0
        r2 = ∂σ_xy/∂x + ∂σ_yy/∂y = 0

    These must hold at every interior point. The loss is the mean
    squared residual over all collocation points:
        L_pde = mean(r1² + r2²)

    Parameters
    ----------
    model          : PINN — the neural network
    xy_collocation : torch.Tensor, shape (N_c, 2), requires_grad=True
                     Random interior points — no labels needed
    E              : float or nn.Parameter — Young's modulus
    nu             : float or nn.Parameter — Poisson's ratio

    Returns
    -------
    torch.Tensor — scalar PDE loss value
    """
    # Forward pass — predict displacement at collocation points
    uv = model(xy_collocation)
    u_x = uv[:, 0:1]
    u_y = uv[:, 1:2]

    # Strain from displacement via autograd
    eps_xx, eps_yy, eps_xy = compute_strain(u_x, u_y, xy_collocation)

    # Stress from strain via Hooke's law
    sig_xx, sig_yy, sig_xy = compute_stress(eps_xx, eps_yy, eps_xy, E, nu)

    # ── Second derivatives: divergence of stress tensor ───────────────────
    # ∂σ_xx/∂x
    dsig_xx_dx = torch.autograd.grad(
        outputs=sig_xx,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(sig_xx),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    # ∂σ_xy/∂y
    dsig_xy_dy = torch.autograd.grad(
        outputs=sig_xy,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(sig_xy),
        create_graph=True,
        retain_graph=True
    )[0][:, 1:2]

    # ∂σ_xy/∂x
    dsig_xy_dx = torch.autograd.grad(
        outputs=sig_xy,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(sig_xy),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    # ∂σ_yy/∂y
    dsig_yy_dy = torch.autograd.grad(
        outputs=sig_yy,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(sig_yy),
        create_graph=True,
        retain_graph=True
    )[0][:, 1:2]

    # Equilibrium residuals
    r1 = dsig_xx_dx + dsig_xy_dy    # x-momentum: ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    r2 = dsig_xy_dx + dsig_yy_dy    # y-momentum: ∂σ_xy/∂x + ∂σ_yy/∂y = 0

    return torch.mean(r1 ** 2 + r2 ** 2)

# Requirement 2: The displacement must be zero on the clamped wall
def loss_dirichlet(model, xy_left):
    """
    Dirichlet boundary condition loss — enforces zero displacement
    on the clamped left wall (x = 0).

        u_x(0, y) = 0
        u_y(0, y) = 0

    Loss:
        L_dir = mean(u_x² + u_y²)  evaluated on left edge points

    Parameters
    ----------
    model   : PINN
    xy_left : torch.Tensor, shape (N_d, 2)
              Points sampled on the left edge (x = 0)

    Returns
    -------
    torch.Tensor — scalar Dirichlet loss value
    """
    uv = model(xy_left)
    u_x = uv[:, 0:1]
    u_y = uv[:, 1:2]

    return torch.mean(u_x ** 2 + u_y ** 2)

# Requirement 3: The surface tractions must match the applied loads
def loss_neumann(model, xy_right, xy_top, xy_bottom, E, nu, traction=1e6):
    """
    Neumann boundary condition loss — enforces correct surface tractions
    on all three traction boundaries.

    Right edge (x = 1): applied traction σ₀ in x direction
        σ_xx = σ₀,  σ_xy = 0
        → residuals: (σ_xx - σ₀)² + σ_xy²

    Top edge (y = 0.5) and bottom edge (y = 0): traction-free
        σ_yy = 0,  σ_xy = 0
        → residuals: σ_yy² + σ_xy²

    Loss:
        L_neu = mean of all traction residuals squared

    Parameters
    ----------
    model      : PINN
    xy_right   : torch.Tensor, shape (N_r, 2)  — points on right edge
    xy_top     : torch.Tensor, shape (N_t, 2)  — points on top edge
    xy_bottom  : torch.Tensor, shape (N_b, 2)  — points on bottom edge
    E, nu      : material parameters
    traction   : float — applied traction magnitude in Pa (default 1 MPa)

    Returns
    -------
    torch.Tensor — scalar Neumann loss value
    """
    def traction_residuals_right(xy):
        uv = model(xy)
        u_x, u_y = uv[:, 0:1], uv[:, 1:2]
        eps_xx, eps_yy, eps_xy = compute_strain(u_x, u_y, xy)
        sig_xx, sig_yy, sig_xy = compute_stress(eps_xx, eps_yy, eps_xy, E, nu)
        traction_norm = traction / S_SCALE   # normalised traction = 1.0
        return torch.mean((sig_xx - traction_norm) ** 2 + sig_xy ** 2)  

    def traction_residuals_free(xy):
        uv = model(xy)
        u_x, u_y = uv[:, 0:1], uv[:, 1:2]
        eps_xx, eps_yy, eps_xy = compute_strain(u_x, u_y, xy)
        sig_xx, sig_yy, sig_xy = compute_stress(eps_xx, eps_yy, eps_xy, E, nu)
        return torch.mean(sig_yy ** 2 + sig_xy ** 2)

    return (traction_residuals_right(xy_right) +
            traction_residuals_free(xy_top) +
            traction_residuals_free(xy_bottom))

# Requirement 4: The predicted displacements at sensor locations must match the observed data
# ── Data-fit loss — inverse mode only ────────────────────────────────────────
# This term is only meaningful when material parameters E and nu are unknown
# and declared as nn.Parameter objects (inverse mode).
#
# In forward mode, E and nu are fixed known values. The three
# physics-based terms (PDE, Dirichlet, Neumann) are sufficient to uniquely
# determine the displacement field — no observations needed.
#
# In inverse mode, the PDE and BC losses alone cannot identify
# E and nu. The equilibrium equations are satisfied by infinitely many
# (field, material) combinations for a homogeneous material. This data term
# breaks that degeneracy — among all physically admissible fields, it selects
# the one consistent with the 20 sparse sensor observations, which uniquely
# pins down E and nu.
#
# Adding this term in forward mode would be counterproductive — it would
# force the network to match one specific FEniCS solution rather than freely
# finding the displacement field that satisfies the physics alone.
def loss_data(model, xy_sensors, u_obs):
    """
    Data-fit loss — used in inverse mode only.

    Penalises the difference between the network's displacement prediction
    at the sensor locations and the FEniCS-computed observations.

        L_data = mean((u_x_pred - u_x_obs)² + (u_y_pred - u_y_obs)²)

    This is the only loss term that provides gradient information about
    E and nu. The PDE and BC losses enforce physical consistency;
    this term identifies which physically consistent solution matches
    the observed data, driving E and nu to their true values.

    Parameters
    ----------
    model      : PINN
    xy_sensors : torch.Tensor, shape (M, 2)  — sensor coordinates
    u_obs      : torch.Tensor, shape (M, 2)  — observed displacements
                 Column 0 is u_x_obs, column 1 is u_y_obs

    Returns
    -------
    torch.Tensor — scalar data-fit loss value
    """
    u_obs_norm = u_obs / U_SCALE    # normalise observations to match network output scale
    uv_pred    = model(xy_sensors)
    return torch.mean((uv_pred - u_obs_norm) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# TOTAL LOSS
# ═══════════════════════════════════════════════════════════════════════════

def total_loss(model, xy_collocation, xy_left, xy_right, xy_top, xy_bottom,
               E, nu, traction=1e6,
               xy_sensors=None, u_obs=None,
               lambda_pde=1.0, lambda_dir=10.0,
               lambda_neu=1.0, lambda_data=10.0):
    """
    Computes the weighted total loss:

        L = λ_pde  * L_pde
          + λ_dir  * L_dir
          + λ_neu  * L_neu
          + λ_data * L_data   (only if xy_sensors and u_obs are provided)

    The lambda weights balance the contribution of each term. Different
    terms have different physical units and magnitudes — without weighting,
    one term can dominate and the others are effectively ignored.

    Default weights: λ_pde=1, λ_dir=10, λ_neu=1, λ_data=10
    The higher weight on Dirichlet and data terms reflects their importance
    in anchoring the solution — these will be tuned during Phase 2.

    Parameters
    ----------
    model            : PINN
    xy_collocation   : interior collocation points, shape (N_c, 2)
    xy_left          : left edge points for Dirichlet BC, shape (N_d, 2)
    xy_right         : right edge points for Neumann BC, shape (N_r, 2)
    xy_top           : top edge points for Neumann BC, shape (N_t, 2)
    xy_bottom        : bottom edge points for Neumann BC, shape (N_b, 2)
    E, nu            : material parameters (float or nn.Parameter)
    traction         : applied traction in Pa
    xy_sensors       : sensor coordinates for data-fit loss (optional)
    u_obs            : observed displacements for data-fit loss (optional)
    lambda_*         : loss weight hyperparameters

    Returns
    -------
    loss_total : torch.Tensor — scalar total loss
    loss_parts : dict — individual loss components for logging
    """
    L_pde = loss_pde(model, xy_collocation, E, nu)
    L_dir = loss_dirichlet(model, xy_left)
    L_neu = loss_neumann(model, xy_right, xy_top, xy_bottom, E, nu, traction)

    loss_total = lambda_pde * L_pde + lambda_dir * L_dir + lambda_neu * L_neu

    loss_parts = {
        "L_pde": L_pde.item(),
        "L_dir": L_dir.item(),
        "L_neu": L_neu.item(),
    }

    # Data-fit term — only active in inverse mode
    if xy_sensors is not None and u_obs is not None:
        L_dat = loss_data(model, xy_sensors, u_obs)
        loss_total = loss_total + lambda_data * L_dat
        loss_parts["L_data"] = L_dat.item()

    loss_parts["L_total"] = loss_total.item()

    return loss_total, loss_parts