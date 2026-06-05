
import numpy as np
from copy import deepcopy
from Capstone.Geometry import circumference, normals, pol2cart


def window_intersections(X, Y, Nx, Ny, window_size=11, step=1, tol=0.1, forward_only=True, eps=1e-12):
    """Find intersections within sliding windows.
    For each window start, test the first ray (index i=start) against all following rays in the window.
    Return compact arrays of hits (i, j, ti, tj, Px, Py)."""

    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    Nx = np.asarray(Nx).ravel()
    Ny = np.asarray(Ny).ravel()
    if not (X.size == Y.size == Nx.size == Ny.size):
        raise ValueError('All inputs must have same length')
    n = X.size
    if window_size < 2:
        raise ValueError('window_size must be >= 2')
    # Prepare arrays
    c = np.stack((X, Y), axis=1)  # (n,2)
    v = np.stack((Nx, Ny), axis=1)

    hits_i = []
    hits_j = []
    hits_ti = []
    hits_tj = []
    hits_P = []
    hits_Py = []
    classes = []
    condis = []
    condjs = []
    filt = np.zeros(n).astype(bool)
    iis = []

    # Helper cross product for arrays
    # def cross2_arr(a_x, a_y, b_x, b_y):
    #     return a_x * b_y - a_y * b_x
    # Slide window (simple Python loop over windows; per-window ops are vectorized)

    print("points to do:",  n - window_size + 1)
    
    for i in range(0, n - window_size + 1, step):
        this = i + int(window_size/2)
        c0 = c[this]            # (2,)
        v0 = v[this]            # (2,)
        c_block = c[i+1:i+window_size]    # (m,2)
        v_block = v[i+1:i+window_size]    # (m,2)
        # Cx = c_block[:,0] - c0[0]
        # Cy = c_block[:,1] - c0[1]
        C = c_block - c0
        # den = cross(v0, v_block)
        # den = cross2_arr(v0[0], v0[1], v_block[:,0], v_block[:,1])
        den = np.cross(v0,v_block)
        # numerators
        # num_ti = cross2_arr(Cx, Cy, v_block[:,0], v_block[:,1])
        num_ti = np.cross(C, v_block)
        # num_tj = cross2_arr(Cx, Cy, v0[0], v0[1])
        num_tj = np.cross(C, v0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ti = num_ti / den
            tj = num_tj / den

        valid = den != 0 #& (np.abs(den) > eps)
        valid = valid & (((-3 < ti) & (ti  < tol*3)) | ((-3 < tj ) & (tj < tol*3)))

        condi = (0 < ti) & (ti < tol*.9)
        condj = (0 < tj) & (tj < tol)


        # these must be none true
        outer = all( condi== False) #| all( condj == False )

        clas = valid*1 + condi*1 + condj*1

            
        # filt[this] = (not any((condi) & (condi != condj))) #| outer
        
        filt[this] = (not any((condi) & (condi != condj))) & (not all( condi== False))

        #filt[this] = outer
        #valid &= cond
        # if not np.any(valid):
        #     continue

        # if filt.sum() > 0:
        #     print('wait', i, end='')

        # compute intersection points for valid entries
        # vi_x = v0[0]; 
        # vi_y = v0[1]
        # Px = c0[0] + ti * vi_x
        # Py = c0[1] + ti * vi_y

        P = c0 + np.stack((ti,ti), axis=1) * v0

        # append hits
        # for valid in np.nonzero(valid)[0]:
        # hits_i.append(i)
        # hits_j.append(i + 1 + int(idx_local))

        iis.append(i)
        hits_ti.append(ti[valid])
        hits_tj.append(tj[valid])
        hits_P.append(P[valid])
        classes.append(clas[valid])
        condis.append(condi[valid])
        condjs.append(condj[valid])

    res = {
            "iis":     np.array(iis),
            # hits_i: hits_i,
            # hits_j: hits_j,
            "hits_ti": np.concatenate(hits_ti), 
            "hits_tj": np.concatenate(hits_tj), 
            "hits_P":  np.concatenate(hits_P), 
            "classes": np.concatenate(classes), 
            "condis":  np.concatenate(condis), 
            "condjs":  np.concatenate(condjs), 
            "filt":    np.array(filt),
        }
    return res



def curve_intersections(X, Y, Nx, Ny, window_size=11, step=1, tol=0.1, forward_only=True, eps=1e-12):
    """Find intersections within sliding windows.
    For each window start, test the first ray (index i=start) against all following rays in the window.
    Return compact arrays of hits (i, j, ti, tj, Px, Py)."""

    # I = np.asarray(I).ravel()
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    Nx = np.asarray(Nx).ravel()
    Ny = np.asarray(Ny).ravel()
    if not (X.size == Y.size == Nx.size == Ny.size):
        raise ValueError('All inputs must have same length')
    n = X.size
    if window_size < 2:
        raise ValueError('window_size must be >= 2')
    # Prepare arrays
    c = np.stack((X, Y), axis=1)  # (n,2)
    # v = np.stack((Nx, Ny), axis=1)
    c1 = np.concatenate((c[-2:,:],c[:-2] ), axis=0)

    v = c1 - c

    hits_ti = []
    hits_tj = []
    hits_P = []
    classes = []
    condis = []
    condjs = []
    filt = np.zeros(n).astype(bool)
    iis = []

    # print("points to do:",  n - window_size + 1)
    
    for i in range(0, n - window_size + 1, step):
        this = i #+ int(window_size/2)
        c0 = c[this]            # (2,)
        v0 = v[this]            # (2,)
        c_block = c[i+2:i+window_size]    # (m,2)
        v_block = v[i+2:i+window_size]    # (m,2)

        C = c_block - c0

        den = np.cross(v0,v_block)

        num_ti = np.cross(C, v_block)

        num_tj = np.cross(C, v0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ti = num_ti / den
            tj = num_tj / den

        l, u = -0.1, 1.1
        valid = den != 0 
        # valid = valid & (((l < ti) & (ti  < u)) | ((l < tj ) & (tj < u)))

        condi = (0 <= ti) & (ti < 1)
        condj = (0 <= tj) & (tj < 1)

        valid = valid & condi & condj

        # these must be none true
        #outer = all( condi== False) #| all( condj == False )

        clas = valid*1 + condi*1 * condj*1

        if any(condi & condj):
            j = i + 2 + np.min(np.where(condi & condj))
            filt[i:j] = True

        # filt[this] = (not any((condi) & (condi != condj))) & (not all( condi== False))
        #filt[this] = any(condi & condj)


        P = c0 + np.stack((ti,ti), axis=1) * v0

        iis.append(i)
        hits_ti.append(ti[valid])
        hits_tj.append(tj[valid])
        hits_P.append(P[valid])
        classes.append(clas[valid])
        condis.append(condi[valid])
        condjs.append(condj[valid])

    res = {
            "iis":     np.array(iis),
            "hits_ti": np.concatenate(hits_ti), 
            "hits_tj": np.concatenate(hits_tj), 
            "hits_P":  np.concatenate(hits_P), 
            "classes": np.concatenate(classes), 
            "condis":  np.concatenate(condis), 
            "condjs":  np.concatenate(condjs), 
            "filt":    np.array(filt),
        }
    return res




def step(I, X, Y, d, s):
    Nx, Ny = normals(X, Y)

    Ex, Ey =  X + d * Nx , Y + d * Ny

    res = curve_intersections(Ex, Ey, Nx, Ny, window_size=50, step=1, tol=d) # *(1+s*0.1)
    
    #res = window_intersections(Ex, Ey, Nx, Ny, window_size=50, step=1, tol=d) # *(1+s*0.1)


    iis, ti_w, tj_w, P, clss, condi, condj, filt = res.values()
    # filt is True where the points should be filtered out / dropped
    
    #Px, Py, ti, tj, valid = adjacent_intersections(Ex, Ey, Nx, Ny, forward_only=True)
    #filt =  np.concatenate([i_idx , j_idx])  #myfilter(ti,tj,d) # & np.append(valid, False)
    
    # if filt.size != 0:
    #     Ex[filt] = np.nan
    #     Ey[filt] = np.nan

    if filt.size != 0:
    #     Ex[filt] = np.nan
    #     Ey[filt] = np.nan


        X1 = Ex[~filt]
        Y1 = Ey[~filt]
        I1 =  I[~filt]


    res = { 'I':I,
        'step': s,
        # 'Theta': T, 
        # 'ThetaDeg':TD, 
        # 'Radius': R, 
        # 'Gradient': GP,
        'X': X,
        'Y': Y,
        'Nx': Nx,
        'Ny': Ny,
        'Ex': Ex,
        'Ey': Ey,
        'I1': I1,
        'X1': X1,
        'Y1': Y1,
        'filt': filt,
        'circ': circumference(X,Y),
        'iis': iis
        }
    extra =  { 'step': s,
        'Px': P[:,0],
        'Py': P[:,1],
        'ti': ti_w,
        'tj': tj_w,
        'condi': condi, 
        'condj': condj,
        'clss' : clss
        # 'valid': np.append(valid, False),
        }


    return I1, X1, Y1, res, extra



def run(func, d = .011, steps = 1, n = 1000 ):
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

    # Theta (radians)
    T = np.linspace(0, np.pi*2 , n)
    I = np.arange(len(T))
    # Calculate Radius for each Theta
    R = func(T)

    X, Y = pol2cart(R, T)

    data = []
    extras = []

    for s in range(steps):
        print("step:", s, "points:", X.shape)

        I, X, Y,res, extra = step(I, X, Y, d, s)

        data.append(deepcopy(res))
        extras.append(deepcopy(extra))


    return data, extras

