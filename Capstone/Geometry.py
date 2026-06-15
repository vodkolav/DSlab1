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

    Gy = np.gradient(Y, X)

    #Check if any gradients got inf or nan
    #assert sum(np.isnan(Gy)) == 0

    #Gx = np.gradient(X, Y) # this should give us an array of 1s, since the gradient of X with respect to itself is 1.

    Gx = np.ones_like(Gy) # not sure this is correct btw.
    # Gx are all 1s assumes the curve Y is a function of X, which is an axis with uniform spacing.
    # But in our case, the X and Y are both functions of theta, since they are generated from a polar coordinate system,
    # So, the assumption may not hold, and the normals may not be accurate. 
    # And that is probably why in some regions the normals don't look perpendicular to the curve in the plot.

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
