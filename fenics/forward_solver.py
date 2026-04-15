import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    TrialFunction, TestFunction, inner, sym, grad,
    Identity, tr, dx, ds, Measure, FacetNormal
)

def create_mesh():
    """
    Creates a rectangular mesh representing the 2D elastic domain.
    Domain: 0 to 1.0 m in x, 0 to 0.5 m in y
    """
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,                # tells FEniCS to use all available processors
        points=[(0.0, 0.0), (1.0, 0.5)],    # defines the corners of the rectangle: (0,0) is the bottom-left corner and (1.0, 0.5) is the top-right corner
        n=[40, 20],                         # how many divisions to make along each axis. 40 divisions in x and 20 divisions in y gives us 40×20 = 800 rectangles
        cell_type=mesh.CellType.triangle    # each rectangle split into 2 triangles, for a total of 1600 triangular elements
    )
    return domain

def create_function_space(domain):
    """
    Creates the finite element function space on the mesh.
    Uses continuous Lagrange elements of degree 1 (linear shape functions). -> Common element type for elasticity problems.
    The elements where the degrees of freedom are the nodal values and the shape functions interpolate between them
    Vector-valued space since we are solving for (u_x, u_y).
    """
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))    # 1 -> linear shape functions, (2,) -> Each node has two degrees of freedom, one for each displacement component
    return V

def apply_boundary_conditions(domain, V):
    """
    Applies the Dirichlet boundary condition on the left edge.
    Left edge (x = 0): u_x = 0 and u_y = 0 (fully clamped wall)
    """
    def left_edge(x):
        return np.isclose(x[0], 0.0)                            # geometric marker

    clamped_dofs = fem.locate_dofs_geometrical(V, left_edge)    # all the displacement unknowns sitting on the left edge
    
    # Dirichlet boundary conditions: on the clamped left wall (displacement is 0)
    bc = fem.dirichletbc(                                       # creates the actual boundary condition object
        np.zeros(2, dtype=np.float64),                          # zero for both u_x and u_y since wall is fully clamped
        clamped_dofs,                                           # which DOFs to apply this condition to
        V                                                       # the function space these DOFs belong to
    )
    return bc

def define_weak_form(domain, V, E, nu, traction=1e6):
    """
    Defines the bilinear and linear forms for linear elasticity.
    Uses the weak form (principle of virtual work):
    
    a(u,v) = integral of sigma(u):epsilon(v) dΩ  (internal virtual work)
    L(v)   = integral of t·v dΓ_right             (external virtual work)
    
    E        -- Young's modulus in Pa
    nu       -- Poisson's ratio (dimensionless)
    traction -- applied horizontal force per unit area on right edge in Pa
    """
    # ── Lamé constants ───────────────────────────────────────────────────────
    # Mathematical reparametrization of the material where Hooke's law can be written in a more compact form. 
    # They are functions of E and nu.
    # or equivalently in terms of λ and μ -> This is cleaner to write than the 3D tensor notation of Hooke's law.
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    # ── Trial and test functions ─────────────────────────────────────────────
    # FEniCS needs both declared explicitly to assemble the stiffness matrix
    u = TrialFunction(V) # The trial function u is what is being solved for: the unknown displacement field
    v = TestFunction(V)  # he test function v is the virtual displacement from the principle of virtual work — a mathematical tool used to convert the PDE into its integral weak form

    # ── Kinematics: strain from displacement ─────────────────────────────────
    def epsilon(u): # Strain tensor
        return sym(grad(u))

    # ── Constitutive law: stress from strain (Hooke's law) ───────────────────
    def sigma(u): # Stress tensor 
        return lam * tr(epsilon(u)) * Identity(2) + 2 * mu * epsilon(u) # This is Hooke's law written in Lamae form 

    # ── Boundary marker: identify the right edge ─────────────────────────────
    # This section identifies which mesh faces sit on the right edge where the traction is applied
    boundaries = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        lambda x: np.isclose(x[0], 1.0)
    )
    boundary_tag = 1
    facet_tags = mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        boundaries,
        np.full(len(boundaries), boundary_tag, dtype=np.int32)
    )
    ds_right = Measure("ds", domain=domain, subdomain_data=facet_tags)(boundary_tag)

    # ── Weak form ────────────────────────────────────────────────────────────
    t = fem.Constant(domain, np.array([traction, 0.0])) # Traction vector: horizontal force of magnitude 'traction' in x-direction, no vertical component
    a = inner(sigma(u), epsilon(v)) * dx                # Bilinear form aka internal virtual work: how the material resists deformation
    L = inner(t, v) * ds_right                          # Linear form aka external virtual work: how the applied traction tries to deform the material
    # When FEniCS assembles these, a becomes the stiffness matrix K and L becomes the load vector f. 
    # Solving Ku = f gives the displacement field.
    return a, L

# This function is where the actual solving happens
# This function takes everything setup previously and computes the solution
def solve_elasticity(domain, V, bc, a, L):
    """
    Solves the linear elasticity problem Ku = f.
    Takes the assembled weak form and boundary condition,
    returns the displacement field as a FEniCS Function.
    """
    problem = LinearProblem(        # FEniCS wrapper that takes your weak form and turns it into a linear system Ku = f and solves it
        a, L,                       # a -> stiffness matrix K, L -> load vector f
        bcs=[bc],                   # Boundary conditions to apply
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"         # LU factorisation is a direct solver: it decomposes K into a lower and upper triangular matrix and solves exactly.
        },
        petsc_options_prefix="elasticity_" # prefix for PETSc options to avoid conflicts if we have multiple solvers in the same code
    )
    # Displacement field uh
    uh = problem.solve() # the discrete finite element approximation of u (as opposed to the exact solution of the PDE u)
    return uh            # ----> The main output of the forward solver. Displacement Fields!


# This function bridges FEniCS and numpy
# it takes the FEniCS solution uh and pulls out displacement values at 
# specific coordinates as plain numpy arrays
def extract_displacement(domain, uh, grid_points, sensor_points):
    """
    Evaluates the FEniCS displacement solution at two sets of points:
    - grid_points  : a dense regular grid of points for full field storage
    - sensor_points: a small set of sparse random points simulating sensors
    
    Returns numpy arrays of displacement values at both sets of points.
    """
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    import numpy as np

    def evaluate_at_points(points):
        # FEniCS expects points as (N, 3) array even in 2D — third column is zero
        points_3d = np.zeros((len(points), 3))
        points_3d[:, :2] = points

        # Build a bounding box tree for efficient point location
        tree = bb_tree(domain, domain.topology.dim)

        # Find which cells (elements) contain each point
        cell_candidates = compute_collisions_points(tree, points_3d)
        colliding_cells = compute_colliding_cells(
            domain, cell_candidates, points_3d
        )

        # Evaluate the solution at each point using the containing cell
        values = []
        cells = []
        points_to_eval = []

        for i, point in enumerate(points_3d):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
                points_to_eval.append(point)

        points_to_eval = np.array(points_to_eval)
        u_values = uh.eval(points_to_eval, cells)
        return u_values  # shape (N, 2) — columns are u_x and u_y

    u_grid    = evaluate_at_points(grid_points)         # dense grid (which gives us the full field for validation later)
    u_sensors = evaluate_at_points(sensor_points)       # sparse sensor points (which gives us the "observations" that the inverse solver will use for training)

    return u_grid, u_sensors


def run_simulation(E, nu, traction=1e6):
    """
    Runs a single forward simulation for a given E and nu.
    
    Parameters
    ----------
    E        : Young's modulus in Pa (e.g. 200e9 for steel)
    nu       : Poisson's ratio (e.g. 0.3)
    traction : applied horizontal traction on right edge in Pa (default 1 MPa)
    
    Returns
    -------
    u_grid    : displacement field on dense grid, shape (1000, 2)
    u_sensors : displacement at sensor locations, shape (20, 2)
    grid_points   : coordinates of dense grid points, shape (1000, 2)
    sensor_points : coordinates of sensor points, shape (20, 2)
    """
    # ── Build mesh and function space ────────────────────────────────────────
    domain = create_mesh()
    V      = create_function_space(domain)

    # ── Boundary conditions ──────────────────────────────────────────────────
    bc = apply_boundary_conditions(domain, V)

    # ── Weak form ────────────────────────────────────────────────────────────
    a, L = define_weak_form(domain, V, E, nu, traction)

    # ── Solve ────────────────────────────────────────────────────────────────
    uh = solve_elasticity(domain, V, bc, a, L)

    # ── Define evaluation points ─────────────────────────────────────────────
    # Dense regular grid: 40 points in x, 25 points in y = 1000 points total
    x_vals = np.linspace(0.01, 0.99, 40)
    y_vals = np.linspace(0.01, 0.49, 25)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Sparse sensor points: 20 random locations fixed by a seed for reproducibility
    rng = np.random.default_rng(seed=42)
    sensor_points = rng.uniform(
        low  = [0.05, 0.05],
        high = [0.95, 0.45],
        size = (20, 2)
    )

    # ── Extract displacements ────────────────────────────────────────────────
    u_grid, u_sensors = extract_displacement(domain, uh, grid_points, sensor_points)

    return u_grid, u_sensors, grid_points, sensor_points

# Code inside this block only runs when you execute the file directly
# it does not run when the file is imported by another script
if __name__ == "__main__":
    # Test with steel-like properties: E = 200 GPa, nu = 0.3
    E_test  = 200e9
    nu_test = 0.3

    print(f"Running forward simulation with E={E_test:.2e} Pa, nu={nu_test}")

    u_grid, u_sensors, grid_points, sensor_points = run_simulation(E_test, nu_test)

    print(f"Grid displacement field shape    : {u_grid.shape}")
    print(f"Sensor displacement values shape : {u_sensors.shape}")
    print(f"Max u_x on grid                  : {u_grid[:, 0].max():.6e} m")
    print(f"Max u_y on grid                  : {u_grid[:, 1].max():.6e} m")

    # Quick analytical check: u_x_max should be approximately traction * L / E
    traction = 1e6
    L        = 1.0
    u_x_analytical = (traction * L) / E_test
    print(f"Analytical u_x estimate          : {u_x_analytical:.6e} m")
    print(f"Difference                       : {abs(u_grid[:, 0].max() - u_x_analytical):.6e} m")