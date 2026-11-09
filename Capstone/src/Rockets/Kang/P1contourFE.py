### P1contourFE.py  

# Contour of P1 FE interpolation

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def N1(xi, eta):
    return 1 - xi - eta

@njit
def N2(xi, eta):
    return xi

@njit
def N3(xi, eta):
    return eta

@njit
def ref_to_real(xi, eta, x_verts, y_verts):
    x = N1(xi, eta)*x_verts[0] + N2(xi, eta)*x_verts[1]
    + N3(xi, eta)*x_verts[2]
    y = N1(xi, eta)*y_verts[0] + N2(xi, eta)*y_verts[1]
    + N3(xi, eta)*y_verts[2]
    return x, y


@njit
def real_to_ref(x, y, x_verts, y_verts):
    A2 = (
        (x_verts[1] - x_verts[0])*(y_verts[2] -
            y_verts[0])
        - (x_verts[2] - x_verts[0])*(y_verts[1] -
            y_verts[0])
    )
    T = np.array([
        [y_verts[2] - y_verts[0], x_verts[0] - x_verts
            [2]],
        [y_verts[0] - y_verts[1], x_verts[1] - x_verts
            [0]]
    ]) / A2
    xy = np.array([
        x - x_verts[0],
        y - y_verts[0]
    ])
    return T@xy

@njit
def element_contour_P1(val, W, x_verts, y_verts, tol=1e-6):
    xi_12 = np.nan
    eta_12 = 0.0

    xi_13 = 0.0
    eta_13 = np.nan

    xi_23 = np.nan
    eta_23 = np.nan

    xi_1 = np.nan
    eta_1 = np.nan

    xi_2 = np.nan
    eta_2 = np.nan

    contour_length = 0.0



    x1 = np.nan
    y1 = np.nan
    x2 = np.nan
    y2 = np.nan

    all_done = False

    if abs(W[1] - W[0]) > tol:
        xi_12 = (val - W[0])/(W[1] - W[0])
    else:
        if abs(val - W[0]) < tol:
            x1 = x_verts[0]
            y1 = y_verts[0]

            x2 = x_verts[1]
            y2 = y_verts[1]

            all_done = True

    if abs(W[2] - W[0]) > tol:
        eta_13 = (val - W[0])/(W[2] - W[0])
    else:
        if abs(val - W[0]) < tol:
            x1 = x_verts[0]
            y1 = y_verts[0]

            x2 = x_verts[2]
            y2 = y_verts[2]

            all_done = True

    if abs(W[1] - W[2]) > tol:
        xi_23 = (val - W[2])/(W[1] - W[2])
        eta_23 = (W[1] - val)/(W[1] - W[2])
    else:
        if abs(val - W[1]) < tol:
            x1 = x_verts[1]
            y1 = y_verts[1]

            x2 = x_verts[2]
            y2 = y_verts[2]

            all_done = True

    if all_done:

        contour_length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    else:
        if xi_12 > 0.0 and xi_12 < 1.0:
            if eta_13 > 0.0 and eta_13 < 1.0:
                xi_1 = xi_12
                eta_1 = eta_12

                xi_2 = xi_13
                eta_2 = eta_13

            elif xi_23 > 0.0 and xi_23 < 1.0:
                xi_1 = xi_12
                eta_1 = eta_12

                xi_2 = xi_23
                eta_2 = eta_23

        elif eta_13 > 0.0 and eta_13 < 1.0:
            if xi_23 > 0.0 and xi_23 < 1.0:
                xi_1 = xi_13
                eta_1 = eta_13

                xi_2 = xi_23
                eta_2 = eta_23

        if not np.isnan(xi_1) and not np.isnan(xi_2):
            x1, y1 = ref_to_real(xi_1, eta_1, x_verts, y_verts)
            x2, y2 = ref_to_real(xi_2, eta_2, x_verts, y_verts)
            contour_length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return contour_length, x1, y1, x2, y2

def extract_contour_P1(levels, nodes, connectivity, field, plot=False):
    contour_lengths = np.zeros_like(levels)
    contour_npoints = np.zeros_like(levels)

    n_elements = np.shape(connectivity)[1] # skfem format

    if plot:
        fig = plt.figure()

    for i_elt in range(n_elements):
        id_nodes = connectivity[:, i_elt]
        x_verts = np.array([nodes[0, k] for k in id_nodes])
        y_verts = np.array([nodes[1, k] for k in id_nodes])
        if plot:
            plt.plot(
                np.hstack((x_verts, np.array([x_verts[0]]))), 
                np.hstack((y_verts, np.array([y_verts[0]]))), 
                'k-', lw=0.5
            )

        for i, level in enumerate(levels):
            l, x1, y1, x2, y2 = element_contour_P1( level, 
                                                   field[id_nodes], 
                                                   x_verts, 
                                                   y_verts )
            contour_lengths[i] += l
            contour_npoints[i] += 1 if l>0 else 0

            if plot and l > 1e-6:
                plt.plot([x1, x2], [y1, y2], 'b-', lw=2)

    if plot:
        plt.show()

    return contour_lengths, contour_npoints

if __name__ == '__main__':
    x_verts = np.array([1, 2, 1.5])
    y_verts = np.array([1, 1.5, 2])
    W = np.array([1, 4, 8])

    l, x1, y1, x2, y2 = element_contour_P1(5.0, W,
        x_verts, y_verts)



    plt.plot(
        np.hstack((x_verts, np.array([x_verts[0]]))), 
        np.hstack((y_verts, np.array([y_verts[0]]))), 
        'k-', lw=2)
    if l > 1e-6:
        plt.plot([x1, x2], [y1, y2], 'b-', lw=2)
    plt.grid()
    plt.show()

    print(1)


