import numpy as np



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def magn(x, y):
    # magnitude of vector
    return np.sqrt(x**2 + y**2)

def rad2deg(rad):
    # convert radians to degrees
    return rad * 180 / np.pi


def circumference(X,Y):
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.diff(X)
        dy = np.diff(Y)
    dxdy = np.sqrt(dx**2 + dy**2)
    C = np.sum(dxdy)
    return C 



def normals(X, Y, ):
    # Calculates normals of the curve (assumes the curve is closed)
    tail = 2
    # Add some samples from opposite ends, 
    # so that X,Y appear circular to gradient
    X = np.concatenate([X[-tail:], X, X[:tail]])
    Y = np.concatenate([Y[-tail:], Y, Y[:tail]])

    Gy = np.gradient(Y)

    #Check if any gradients got inf or nan
    #assert sum(np.isnan(Gy)) == 0

    #Gx = np.gradient(X, Y) # this should give us an array of 1s, since the gradient of X with respect to itself is 1.

    Gx = np.gradient(X)
    # Calculate normals
    L = magn(Gx, Gy)
    Nx, Ny = -Gy/L, Gx/L 
    xy  = np.array([X,Y])
    NxNy = np.array([Nx,Ny])

    # Ensure all normal vectors point in the same 
    # direction (Relative to origin) 
    dot_products = np.sum(xy * NxNy, axis=0)

    scaling_factor_S = np.sign(dot_products)

    S_reshaped = scaling_factor_S[np.newaxis, :]

    N_consistent = NxNy * S_reshaped 
    Nx, Ny =  N_consistent[:, tail:-tail]
    return Nx, Ny


def intersections(c0, v0, c_block, v_block):
    C = c_block - c0

        # denominators
    den = np.cross(v0, v_block)

        # numerators
    num_ti = np.cross(C, v_block)

    num_tj = np.cross(C, v0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ti = num_ti / den
        tj = num_tj / den

    valid = den != 0
    return ti,tj,valid


def cslice(start, stop, n):
    """circular slice. like a clock

    Args:
        n (int): total amt of items in circular array
        start (int): start position of slice
        stop (int): stop position of slice

    Returns:
        list: indices to extract
    """
    res = np.arange(start, stop) % n
    return res

# Convert between Cartesian/Polar coordinates

def arc_lens(X,Y):
    """
    Calculate lengths of curve segments defined by points (X,Y).

    Args:
        X (_type_): _description_
        Y (_type_): _description_

    Returns:
        _type_: _description_
    """

    XY = np.stack((X,Y), axis=1)
    XY = np.concatenate((XY,XY[0:1,:]), axis=0)

    diff = np.diff(XY, axis=0)
    Lens = np.sqrt(np.sum(diff ** 2, axis=1))
    return Lens


def slerp(v1, v2, d):
    """
    Interpolate between two 2D vectors with points spaced along the circular arc.

    Parameters
    ----------
    v1, v2 : array_like, shape (2,)
        Vectors from the same origin and with the same length.
    d : float
        Maximum allowed arc length between adjacent returned points.

    Returns
    -------
    np.ndarray
        Array of shape (k, 2) containing the intermediate points between v1 and v2.
        If d is larger than the arc length, an empty array is returned.
    """
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v2 = np.asarray(v2, dtype=float).reshape(-1)

    if v1.shape != (2,) or v2.shape != (2,):
        raise ValueError("v1 and v2 must each be 2D vectors")
    if d <= 0:
        raise ValueError("d must be positive")

    r1 = np.linalg.norm(v1)
    r2 = np.linalg.norm(v2)
    if np.isclose(r1, 0.0) or np.isclose(r2, 0.0):
        raise ValueError("v1 and v2 must be non-zero vectors")
    # if not np.isclose(r1, r2):
    #     raise ValueError("v1 and v2 must have the same length")

    u1 = v1 / r1
    u2 = v2 / r2

    # Signed angle from v1 to v2 (counter-clockwise if positive)
    theta = np.arctan2(u1[0] * u2[1] - u1[1] * u2[0], np.dot(u1, u2))
    if np.isclose(theta, 0.0):
        return np.empty((0, 2), dtype=float)

    arc_len = r1 * abs(theta)
    if d > arc_len:
        return np.empty((0, 2), dtype=float)

    n_segments = max(1, int(np.ceil(arc_len / d)))
    t = np.linspace(0.0, 1.0, n_segments + 1)[1:-1]
    if t.size == 0:
        return np.empty((0, 2), dtype=float)

    angles = theta * t
    c = np.cos(angles)
    s = np.sin(angles)

    points = np.column_stack((
        c * u1[0] - s * u1[1],
        s * u1[0] + c * u1[1],
    )) * r1

    return points