
import numpy as np
from scipy.interpolate import Rbf, CubicSpline
from Capstone.Geometry import circumference, normals, magn, pol2cart, cart2pol
from Capstone.Rockets.Harvester import Harvester


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


def window_intersections(XY, N, window_size=11, step=1, tol=0.1, forward_only=True, eps=1e-12):
    """Find intersections within sliding windows.
    For each window start, test the first ray (index i=start) against all following rays in the window.
    Return compact arrays of hits (i, j, ti, tj, Px, Py)."""


    n = XY.shape[0]

    # Prepare arrays
    c = XY
    v = N

    filt = np.zeros(n).astype(bool)


    for i in range(0, n - window_size + 1, step):
        this = i + int(window_size/2)
        c0 = c[this]            # (2,)
        v0 = v[this]            # (2,)

        c_block = c[i+1:i+window_size]    # (m,2)
        v_block = v[i+1:i+window_size]    # (m,2)

        ti, tj, valid = intersections(c0, v0, c_block, v_block) 

        valid = valid & (((-3 < ti) & (ti  < tol*3)) | ((-3 < tj ) & (tj < tol*3)))


        condi = (0 < ti) & (ti < tol*.9)
        condj = (0 < tj) & (tj < tol)

        clas = valid*1 + condi*1 + condj*1

        filt[this] = (not any((condi) & (condi != condj))) & (not all( condi== False))

        # compute intersection points for valid entries

        P = c0 + np.stack((ti,ti), axis=1) * v0

        Px = P[:,0]
        Py = P[:,1]
        ii = np.ones_like(ti)*i

    return filt


HSintersections = Harvester(varnames=["SimStep", "ii", "ti", "tj", "Px", "Py",
                                         "clas", "condi", "condj"],
                            elems='valid')


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


def curve_intersections(XY, N, window_size=11, SimStep=1, tol=0.1, forward_only=True, eps=1e-12):
    """Find intersections within sliding windows.
    For each window start, test the first ray (index i=start) against all following rays in the window.
    Return compact arrays of hits (i, j, ti, tj, Px, Py)."""


    n = XY.shape[0]

    c = XY # segment start points

    c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
    v = c1 - c # segment direction vectors

    filt = np.zeros(n).astype(bool)


    for i in range(0, n , 1):

        c0 = c[i]            # (2,)
        v0 = v[i]            # (2,)

        sl = cslice(i+2, i+window_size, n)

        c_block = c[sl] 
        v_block = v[sl] 
        # rest of the vectors in window (shifted by 2 to avoid adjacent segments)
        # Adjacent segments by definition intersect at their shared vertex, which is not a valid intersection for our purposes. 
        # By shifting by 2, we ensure that we are only checking for intersections between non-adjacent segments.


        ti, tj, valid = intersections(c0, v0, c_block, v_block) 
        # valid = valid & (((l < ti) & (ti  < u)) | ((l < tj ) & (tj < u)))

        condi = (0 <= ti) & (ti < 1)
        condj = (0 <= tj) & (tj < 1)

        valid = valid & condi & condj

        clas = valid*1 + condi*1 * condj*1

        if any(condi & condj):
            j = i + 2 + np.min(np.where(condi & condj))
            sl = cslice(i,j,n)
            filt[sl] = True


        P = c0 + np.stack((ti,ti), axis=1) * v0

        Px = P[:,0]
        Py = P[:,1]
        ii = np.ones_like(ti)*i

        HSintersections.collect(locals())

    return filt


# rarefactions: 
def rarefactions(I, X, Y, d = 0.1):
    
    XY = np.stack((X,Y), axis=1)
    # n = XY.shape[0]

    c = XY # segment start points

    c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
    v = c1 - c # segment direction vectors    

    m = magn(v[:,0], v[:,1])

    isNew = np.zeros_like(X, dtype=bool)

    quantiles = np.sum(m[:, None] > m, axis=1) / (len(m) - 1)
    # only take the points in the top 2% of segment lengths, e.g points that diverged the most
    divergents = (m > 2*d) & (quantiles > .98) 

    return divergents


def interpolate(XY, xy):
    x,y = np.split(XY,2, axis=1)
    xi,_ = np.split(xy,2, axis=1)
    # TODO: option to choose interpolation method in simulation settings.
    # rbf = Rbf(x, y)
    rbf = CubicSpline(x.squeeze(), y.squeeze())
    yi = rbf(xi)
    return np.concatenate((xi,yi), axis = 1)


def fill(I, X, Y, divergents, d,  method = 'dumb'):

    XY = np.stack((X,Y), axis=1)
    # n = XY.shape[0]

    c = XY # segment start points

    c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
    v = c1 - c # segment direction vectors    

    if method == 'dumb':
            # simplest interpolation: just add half the segment vector to the start point of the segment
            newXY = XY[divergents] + v[divergents] * 0.5
            newI = I[divergents]
    else:
        # add multiple points along the segment vector
        # direction vectors of divergents
        # hi = XY[divergents+1,:] - XY[divergents,:]
        hi = v[divergents,:]

        l = np.sqrt(np.sum(hi**2,axis=1))
        jj = (l/d).astype(int)       

        dividx = I[divergents]

        # cubXY = np.zeros((0,2))
        newXY = np.zeros((0,2))
        newI = np.zeros(0)

        for i,rr in enumerate(dividx):
            j = np.arange(1, jj[i]).reshape(jj[i]-1,1) 

            wat = np.dot(j,hi[[i],:])
            
            xy = XY[rr,:] + wat / jj[i]

            if method == 'interp':
                slc = cslice(rr-3,rr+3, XY.shape[0])
                xy = interpolate(XY[slc,:], xy)


            newXY = np.concatenate((newXY, xy), axis=0)
            onns = np.ones(jj[i]-1)

            newI = np.concatenate((newI, onns*(rr)), axis=0)
            # print("a", a)
        newI = newI.astype(int)
        newI, newXY

    newX, newY =  newXY[:,0], newXY[:,1]
    return newI, newX, newY


def active(X,Y,rh):
    R,_ = cart2pol(X,Y)
    A = R < rh
    return A


HScurves = Harvester(varnames=["I", "X", "Y", "A", "S", "isNew",
                               "Nx", "Ny", "Ex", "Ey",
                               "filt"], on_size_mismatch='error')
                              #"I1","X1", "Y1", circ, , "isNew" 

def step(I, X, Y, A, isNew, d, rh, s, window_size=50):
    Nx, Ny = normals(X, Y)

    Ex, Ey =  X + d * A * Nx , Y + d * A * Ny

        # I = np.asarray(I).ravel()
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    Nx = np.asarray(Nx).ravel()
    Ny = np.asarray(Ny).ravel()
    if not (X.size == Y.size == Nx.size == Ny.size):
        raise ValueError('All inputs must have same length')
    
    if window_size < 2:
        raise ValueError('window_size must be >= 2')
    # Prepare arrays
    E = np.stack((Ex, Ey), axis=1)  # (n,2)
    N = np.stack((Nx, Ny), axis=1)  # (n,2)


    filt = curve_intersections(E, N, window_size=window_size, SimStep=s, tol=d) # *(1+s*0.1)
    
    #res = window_intersections(E, N, window_size=50, step=1, tol=d) # *(1+s*0.1)
    #Px, Py, ti, tj, valid = adjacent_intersections(Ex, Ey, Nx, Ny, forward_only=True)

    # filt is True where the points should be filtered out / dropped

    if filt.size != 0:
        X1 = Ex[~filt]
        Y1 = Ey[~filt]
        I1 = np.arange(X1.shape[0])

    divergents = rarefactions(I1, X1, Y1)

    newI, newX, newY =  fill(I1, X1, Y1, divergents, d, method = 'manydumb')

    news = np.ones_like(newI)
    isNew1 = np.zeros_like(X1)


    # TODO: make this a single array operation instead of 3 separate ones
    X1 = np.insert(X1, newI[:-1], newX[:-1], axis=0)
    Y1 = np.insert(Y1, newI[:-1], newY[:-1], axis=0)
    I1 = np.arange(X1.shape[0])
    isNew1 = np.insert(isNew1, newI[:-1], news[:-1], axis=0)


    A1 = active(X1,Y1, rh)

    S = np.ones_like(A)*s
    C = circumference(X1*A1,Y1*A1)

    HScurves.collect(locals())

    return I1, X1, Y1, A1, isNew1, C


HSsim = Harvester(["s", "C"])

def grain(func, n):
    T = np.linspace(0, np.pi*2, n, endpoint=False) - 0.0001
    # T = np.append(T, T[:1])
    I = np.arange(len(T))
    # Calculate Radius for each Theta
    R = func(T)
    X, Y = pol2cart(R, T)
    return I, R, T, X, Y 


def run(func, d = .011, steps = 1, n = 1000, hr = 4 ):
    """runs the simulation

    Args:
        func (def): function defining the curve. must be of form Radius = f(Theta)
        d (float, optional): step size. Defaults to -.3.
        steps (int, optional): simulation steps. Defaults to 1.
        n (int, optional): simulation points. Defaults to 1000.

    Returns:
        dict: dictionary of results data
    """
    
    print('Simulation step size(d):',d)
    print('Simulation Steps:', steps)
    print('Curve points (n):', n)

    I, R, T, X, Y = grain(func, n)
    #hr = 4 # hull radius
    Hr = np.ones_like(T)*hr
    Hx, Hy = pol2cart(Hr, T)
    A = active(X,Y, hr)

    IsNew = np.zeros_like(I)

    for s in range(steps):
        print("step:", s, "points:", X.shape)

        I, X, Y, A, IsNew, C = step(I, X, Y, A, IsNew, d, hr, s)

        HSsim.collect(locals())
        if sum(A) == 0:
            print("everything's burnt")
            break

    HSsim.add_once(Hx = Hx, Hy = Hy)



