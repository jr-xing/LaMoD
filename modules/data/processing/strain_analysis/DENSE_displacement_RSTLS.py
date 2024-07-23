import numpy as np
from scipy import sparse
from scipy.interpolate import griddata
# Additional imports may be required for specific functionality

def DENSE_displacement_RSTLS(x, y, z, xnodes, ynodes, zf1, **kwargs):
    """
    Translated function from MATLAB to compute Lagrangian displacement field
    from Eulerian displacement field.

    Parameters:
    x, y, z: Arrays of data points
    xnodes, ynodes: Grid nodes
    zf1: Additional parameter (similar to MATLAB function)
    kwargs: Additional parameters (similar to varargin in MATLAB)

    Returns:
    zgrid: Computed displacement field
    """

    # Set default parameters and update with kwargs
    params = {
        'smoothness': 4,
        'Tempsmoothness': 0.5,
        'interp': 'bilinear',
        'solver': 'normal',
        'maxiter': [],
        'extend': 'warning',
        'tilesize': np.inf,
        'overlap': 0.2,
        'mask': [],
        'faster': True
    }
    params.update(kwargs)

    # Check parameters (this function needs to be defined)
    params = check_params(params)

    # Ensure inputs are column vectors and remove NaNs
    x, y, z = [np.array(a).ravel() for a in [x, y, z]]
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]

    # Process xnodes and ynodes (similar to MATLAB code)
    xnodes, ynodes = process_nodes(x, y, xnodes, ynodes, params)

    # Main computation (needs translation from MATLAB code)
    nx, ny = len(xnodes), len(ynodes)
    ngrid = nx * ny
    n = len(x)

    # Determine which cell in the grid each point lies in
    indx = np.searchsorted(xnodes, x, side='right') - 1
    indy = np.searchsorted(ynodes, y, side='right') - 1

    # Ensure indices are within bounds
    indx = np.clip(indx, 0, nx - 2)
    indy = np.clip(indy, 0, ny - 2)

    # Interpolation equations for each point
    A, rhs = interpolation_equations(params['interp'], x, y, xnodes, ynodes, indx, indy, z, nx, ny)

    # Regularizer (for smoothness)
    Areg = build_regularizer(xnodes, ynodes, nx, ny, params['smoothness'])

    # Solve the system using A, rhs, and Areg
    solution = solve_system(A, rhs, Areg, params)

    # Reshape the solution to grid format
    zgrid = np.reshape(solution, (ny, nx))

    # Return computed displacement field
    return zgrid

# Helper functions like check_params() need to be defined
# ...

# Example usage:
# zgrid = dense_displacement_RSTLS(x, y, z, xnodes, ynodes, zf1)
def check_params(params):
    """
    Check and validate the parameters.

    Parameters:
    params: Dictionary of parameter values.

    Returns:
    Updated dictionary of parameters after validation.
    """

    # Check 'smoothness' parameter
    if 'smoothness' not in params or params['smoothness'] <= 0:
        raise ValueError("Smoothness must be positive.")

    # Check 'interp' parameter
    valid_interp_methods = ['bicubic', 'bilinear', 'nearest', 'triangle']
    if params.get('interp', '').lower() not in valid_interp_methods:
        raise ValueError(f"Invalid interpolation method: {params.get('interp')}")

    # Check 'solver' parameter
    valid_solver_methods = ['backslash', '\\', 'symmlq', 'lsqr', 'normal']
    if params.get('solver', '').lower() not in valid_solver_methods:
        raise ValueError(f"Invalid solver option: {params.get('solver')}")

    # Check 'extend' parameter
    valid_extend_options = ['never', 'warning', 'always']
    if params.get('extend', '').lower() not in valid_extend_options:
        raise ValueError(f"Invalid extend option: {params.get('extend')}")

    # Check 'tilesize' parameter
    if 'tilesize' in params and (params['tilesize'] < 3 or not isinstance(params['tilesize'], (int, float))):
        raise ValueError("Tilesize must be a positive number greater than or equal to 3.")

    # Check 'overlap' parameter
    if 'overlap' in params and (not 0 < params['overlap'] < 1):
        raise ValueError("Overlap must be a number between 0 and 1.")

    # Add other parameter checks as needed
    # ...

    return params

def process_nodes(x, y, xnodes, ynodes, params):
    """
    Process xnodes and ynodes to create grid nodes.

    Parameters:
    x, y: Input data points.
    xnodes, ynodes: Initial grid nodes.
    params: Dictionary of parameter values.

    Returns:
    Processed xnodes and ynodes.
    """
    # Convert to numpy arrays
    xnodes = np.array(xnodes, dtype=float)
    ynodes = np.array(ynodes, dtype=float)

    # Determine the extent of the data
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Check if xnodes and ynodes are scalar and create linspace if they are
    if xnodes.size == 1:
        xnodes = np.linspace(xmin, xmax, int(xnodes))
        xnodes[-1] = xmax  # Ensure it hits the max
    if ynodes.size == 1:
        ynodes = np.linspace(ymin, ymax, int(ynodes))
        ynodes[-1] = ymax  # Ensure it hits the max

    # Check if nodes are monotone increasing
    if np.any(np.diff(xnodes) <= 0) or np.any(np.diff(ynodes) <= 0):
        raise ValueError("xnodes and ynodes must be monotone increasing")

    # Extend the nodes if necessary, based on params['extend']
    xnodes, ynodes = extend_nodes(xnodes, ynodes, xmin, xmax, ymin, ymax, params)

    return xnodes, ynodes


def extend_nodes(xnodes, ynodes, xmin, xmax, ymin, ymax, params):
    """
    Extend the nodes if the data points fall outside the original grid.

    Parameters:
    xnodes, ynodes: Grid nodes.
    xmin, xmax, ymin, ymax: Data extent.
    params: Dictionary of parameter values.

    Returns:
    Extended xnodes and ynodes.
    """
    # Extend xnodes
    if xmin < xnodes[0]:
        if params['extend'] in ['always', 'warning']:
            if params['extend'] == 'warning':
                print(f"Warning: xnodes(1) was decreased to {xmin}")
            xnodes[0] = xmin
        elif params['extend'] == 'never':
            raise ValueError(f"Some x values fall below xnodes(1) by {xnodes[0] - xmin}")

    if xmax > xnodes[-1]:
        if params['extend'] in ['always', 'warning']:
            if params['extend'] == 'warning':
                print(f"Warning: xnodes(end) was increased to {xmax}")
            xnodes[-1] = xmax
        elif params['extend'] == 'never':
            raise ValueError(f"Some x values fall above xnodes(end) by {xmax - xnodes[-1]}")

    # Extend ynodes
    if ymin < ynodes[0]:
        if params['extend'] in ['always', 'warning']:
            if params['extend'] == 'warning':
                print(f"Warning: ynodes(1) was decreased to {ymin}")
            ynodes[0] = ymin
        elif params['extend'] == 'never':
            raise ValueError(f"Some y values fall below ynodes(1) by {ynodes[0] - ymin}")

    if ymax > ynodes[-1]:
        if params['extend'] in ['always', 'warning']:
            if params['extend'] == 'warning':
                print(f"Warning: ynodes(end) was increased to {ymax}")
            ynodes[-1] = ymax
        elif params['extend'] == 'never':
            raise ValueError(f"Some y values fall above ynodes(end) by {ymax - ynodes[-1]}")

    return xnodes, ynodes

def interpolation_equations(method, x, y, xnodes, ynodes, indx, indy, z, nx, ny):
    """
    Set up interpolation equations based on the specified method.

    Parameters:
    method: Interpolation method ('bilinear', 'triangle', 'nearest').
    x, y: Arrays of data points.
    xnodes, ynodes: Grid nodes.
    indx, indy: Indices of grid cells for each point.
    z: Array of values at data points.
    nx, ny: Number of nodes in x and y.

    Returns:
    A: Sparse matrix for interpolation.
    rhs: Right-hand side vector for the system.
    """
    dx = np.diff(xnodes)
    dy = np.diff(ynodes)

    # Calculate relative positions within each cell
    tx = (x - xnodes[indx]) / dx[indx]
    ty = (y - ynodes[indy]) / dy[indy]

    if method == 'bilinear':
        # Contributions to the nearest four grid points
        contributions = [(1 - tx) * (1 - ty), (1 - tx) * ty, tx * (1 - ty), tx * ty]
        indices = [indy + ny * indx, indy + 1 + ny * indx, indy + ny * (indx + 1), indy + 1 + ny * (indx + 1)]
    
    elif method == 'triangle':
        # Contributions to the three nearest grid points forming a triangle
        k = tx > ty
        L = np.ones_like(tx)
        L[k] = ny
        
        t1 = np.minimum(tx, ty)
        t2 = np.maximum(tx, ty)
        contributions = [1 - t2, t1, t2 - t1]
        indices = [indy + ny * indx, indy + ny * indx + 1, indy + L + ny * indx]

    elif method == 'nearest':
        # Contribution only to the nearest grid point
        k = np.round(1 - ty) + np.round(1 - tx) * ny
        contributions = [np.ones_like(tx)]
        indices = [indy + k + ny * indx]

    # Construct sparse matrix
    rows = np.repeat(np.arange(len(x)), len(contributions))
    cols = np.concatenate(indices)
    data = np.concatenate(contributions)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(len(x), nx * ny))

    # Right-hand side
    rhs = z

    return A, rhs

from scipy.sparse import diags

def build_regularizer(xnodes, ynodes, nx, ny, smoothness):
    """
    Build a regularizer using second derivatives to enforce smoothness.

    Parameters:
    xnodes, ynodes: Grid nodes.
    nx, ny: Number of nodes in x and y.
    smoothness: Smoothness parameter.

    Returns:
    Areg: Sparse matrix representing the regularizer.
    """
    # Second derivative with respect to x
    dx = np.diff(xnodes)
    dxx = diags([1 / dx, -2 / dx[:-1] + -2 / dx[1:], 1 / dx], offsets=[-1, 0, 1], shape=(nx - 1, nx))
    Dx = sparse.kron(dxx, sparse.eye(ny))

    # Second derivative with respect to y
    dy = np.diff(ynodes)
    dyy = diags([1 / dy, -2 / dy[:-1] + -2 / dy[1:], 1 / dy], offsets=[-1, 0, 1], shape=(ny - 1, ny))
    Dy = sparse.kron(sparse.eye(nx), dyy)

    # Combine the two derivatives
    Areg = smoothness * (Dx.T @ Dx + Dy.T @ Dy)

    return Areg

from scipy.sparse.linalg import lsqr, LinearOperator

def solve_system(A, rhs, Areg, params):
    """
    Solve the system using the specified solver in params.

    Parameters:
    A: Sparse matrix for interpolation.
    rhs: Right-hand side vector for the system.
    Areg: Regularizer matrix.
    params: Dictionary of parameter values.

    Returns:
    solution: The solution to the system.
    """
    # Combine A and Areg
    A_combined = sparse.vstack([A, Areg])

    # Update rhs for the regularizer
    rhs_combined = np.hstack([rhs, np.zeros(Areg.shape[0])])

    # Select and apply the solver
    if params['solver'] in ['\\', 'backslash']:
        # Direct solver (use SciPy's sparse.linalg.spsolve for sparse systems)
        solution = sparse.linalg.spsolve(A_combined, rhs_combined)
    
    elif params['solver'] == 'lsqr':
        # LSQR iterative solver
        solution = lsqr(A_combined, rhs_combined)[0]

    elif params['solver'] == 'symmlq':
        # SYMLQ iterative solver
        linear_operator = LinearOperator(A_combined.shape, matvec=lambda x: A_combined @ x)
        solution, _ = symmlq(linear_operator, rhs_combined)

    else:
        raise ValueError(f"Unknown solver: {params['solver']}")

    return solution
