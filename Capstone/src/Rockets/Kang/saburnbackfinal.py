### saburnbackfinal.py  

# Simulated annealing for burnback analysis

import numpy as np
import matplotlib.pyplot as plt
from pyfreefem import FreeFemRunner
from Rockets.Kang.P1contourFE import extract_contour_P1

A_MIN = 0.1
A_MAX = 0.8
B_MIN = 0.1
B_MAX = 0.8
M_MIN = 1
M_MAX = 10
N1_MIN = 0.1
N1_MAX = 15.0
N2_MIN = 0.1
N2_MAX = 15.0
N3_MIN = 0.1
N3_MAX = 15.0

R_A = 0.5
R_B = 0.5
R_M = 1
R_N1 = 1.0
R_N2 = 1.0
R_N3 = 1.0

EQUAL_A_B = True
EQUAL_N23 = True

PC0 = 101325.0
MPA = 1e6

# C_T = 1e5
C_T = 0

edpscript = "Capstone/src/Rockets/Kang/burnbackpftemplate.edp"


def solve_burnback(
    a_sf, b_sf, m_sf, n1_sf, n2_sf, n3_sf,
    nm_o=100, nm_i=100, C_pf=1e8, Ro=1.0, Ri=0.1,
    plot=False
):
    exports = FreeFemRunner(edpscript).execute({
        'M': m_sf,
        'A': a_sf,
        'B': b_sf,
        'N1': n1_sf,
        'N2': n2_sf,
        'N3': n3_sf,
        'NMO': nm_o,
        'NMI': nm_i,
        'CPF': C_pf,
        'RO': Ro,
        'RI': Ri
    })

    mesh = exports['Th']
    S = exports['S[]']

    if plot:
        theta = np.linspace(0, 2*np.pi, 101)
        Ro = 1.0
        xo = Ro * np.cos(theta)
        yo = Ro * np.sin(theta)

        plt.figure()
        plt.plot(xo, yo, 'k-', lw=2)
        plt.tricontour(mesh.x, mesh.y, S, levels=[k*1e-2 for k in range(1,101)])
        # plt.tricontour(mesh.x, mesh.y, S, levels=np.linspace(1e-2, 1, 10), linewidths=2)
        # plt.xlim(-1.25, 1.25)
        # plt.ylim(-1.25, 1.25)
        plt.axis('equal')
        # plt.show()

    return mesh, S


def pressure_profile(mesh, S, DT=1e-2, plot=False,
                     rho_p = 1700.0,
                     c_star = 1400.0,
                     a = 2e-4,
                     n = 0.3,
                     At = 5e-2
                     ):

    sz = np.max(S) - np.min(S)
    n_levels = int(sz/DT) + 1
    web_vals = np.linspace(np.min(S), np.max(S), n_levels)
    Pc_curve = np.zeros_like(web_vals)
    Pc_curve[0] = PC0

    perimeters, npoints = extract_contour_P1(
        web_vals, 
        mesh.vertices[:,:2].T,
        mesh.triangles[:,:3].T - 1,
        S,
        plot=plot
    )

    Pc_curve[1:] = np.power(
        (rho_p * perimeters[1:] * a * c_star / At),
        (1/(1 - n))
    )

    Pc_curve = np.array(Pc_curve)
    Pc_curve = np.maximum(Pc_curve, PC0)
    # Pc_curve[-1] = PC0
    Pc_MPA = np.array(Pc_curve)/MPA

    time_vals = [0]
    for i in range(len(web_vals)-1):
        t_new = 0.5*((1/(a*Pc_curve[i  ]**n))+
                     (1/(a*Pc_curve[i+1]**n)))*(web_vals[i+1]-web_vals[i]) 
        + time_vals[i]
        time_vals.append(t_new)

    if plot:
        plt.figure()
        plt.plot(web_vals, Pc_MPA)
        plt.xlabel("Burned Web Thickness [m]")
        plt.ylabel("Chamber Pressure [MPa]")
        # plt.title("Burned Web Thickness vs Chamber Pressure")
        plt.grid()
        # plt.show()
        plt.figure()
        plt.plot(time_vals, Pc_MPA)
        plt.xlabel("Time [s]")
        plt.ylabel("Chamber Pressure [MPa]")
        # plt.title("Time vs Chamber Pressure")
        plt.grid()
        # plt.show()

    return web_vals, Pc_curve, time_vals, perimeters, npoints


def solve_SA(
    energy,
    x0,
    transition,
    T0,
    T_drop=0.99,
    n_iter=200,
    tol=1e-2,
    verbose=False
    ):
    x_curr = np.zeros_like(x0)
    x_curr[:] = x0
    T_curr = T0
    E_curr = energy(x_curr)
    best_error = 1.0

    list_x = [x_curr]
    list_E = [E_curr]
    list_T = [T_curr]

    iter = 0
    while E_curr > tol and iter < n_iter:
        x_new = transition(x_curr, E_curr)
        E_new = energy(x_new)

        r = np.random.random()
        if r < np.exp(-(E_new - E_curr)/T_curr):
            x_curr[:] = x_new
            E_curr = E_new

        T_curr *= T_drop
        iter += 1

        list_T.append(T_curr)
        list_x.append(x_curr)
        list_E.append(E_curr)

        if E_new < best_error:
            best_error = E_new
            best_x = x_new

        if verbose:
            print(f'Iteration {iter}: T = {T_curr}, x ={x_curr}, E = {E_curr}')

    return x_curr, E_curr, list_T, list_x, list_E, best_error, best_x

def energy_sf(x, ts_ref, Ps_ref, n_div=100):
    mesh, S = solve_burnback(
        x[0],
        x[1],
        int(x[2]),
        x[3],
        x[4],
        x[5])

    ts, Ps, time_vals, perimeters, npoints = pressure_profile(mesh, S)

    t_end = ts[-1]
    t_end_ref = ts_ref[-1]

    if ts[-1] < ts_ref[-1]:
        ts = np.hstack((ts, np.array([ts_ref[-1]])))
        Ps = np.hstack((Ps, np.array([Ps_ref[-1]])))
    elif ts[-1] > ts_ref[-1]:
        ts_ref = np.hstack((ts_ref, np.array([ts[-1]])))
        Ps_ref = np.hstack((Ps_ref, np.array([Ps[-1]])))

    T = ts_ref[-1] - ts_ref[0]
    h = T/n_div
    dp = 0.0

    for i in range(n_div):
        tl = ts_ref[0] + i*h
        tr = ts_ref[0] + (i+1)*h
        Pl = np.interp(tl, ts, Ps)
        Pr = np.interp(tr, ts, Ps)
        Pl_ref = np.interp(tl, ts_ref, Ps_ref)
        Pr_ref = np.interp(tr, ts_ref, Ps_ref)
        dp += 0.5*h*((Pl-Pl_ref)**2 + (Pr - Pr_ref)**2)

    dp = np.sqrt(dp)

    return (dp + C_T * (t_end - t_end_ref)**2)

def transition_sf(x, error, equal_a_b=True, equal_n_23=True):
    x_new = np.zeros_like(x)

    if error ==0.0:
        pass
    elif error >= 1.0:
        R_A = 0.15
        R_N2 = 5.0
        R_N1 = 5.0
        R_M = 1
    elif error > 0.1:
        R_A = 0.1
        R_N2 = 3.0
        R_N1 = 3.0
        R_M = 1
    elif error <= 0.1:
        R_A = 0.05
        R_N2 = 1.0
        R_N1 = 1.0
        R_M = 1

    x_new[0] = x[0] + R_A * np.random.uniform(-1.0, 1.0)
    if x_new[0] < A_MIN:
        x_new[0] = A_MIN
    elif x_new[0] > A_MAX:
        x_new[0] = A_MAX

    if equal_a_b:
        x_new[1] = x_new[0]
    else:
        x_new[1] = x[1] + R_B * np.random.uniform(-1.0, 1.0)
        if x_new[1] < B_MIN:
            x_new[1] = B_MIN
        elif x_new[1] > B_MAX:
            x_new[1] = B_MAX

    x_new[2] = x[2] + np.random.randint(-R_M, R_M + 1)
    if x_new[2] < M_MIN:
        x_new[2] = M_MIN
    elif x_new[2] > M_MAX:
        x_new[2] = M_MAX

    x_new[3] = x[3] + R_N1 * np.random.uniform(-1.0, 1.0)
    if x_new[3] < N1_MIN:
        x_new[3] = N1_MIN
    elif x_new[3] > N1_MAX:
        x_new[3] = N1_MAX

    x_new[4] = x[4] + R_N2 * np.random.uniform(-1.0, 1.0)
    if x_new[4] < N2_MIN:
        x_new[4] = N2_MIN
    elif x_new[4] > N2_MAX:
        x_new[4] = N2_MAX

    if equal_n_23:
        x_new[5] = x_new[4]
    else:
        x_new[5] = x[5] + R_N3 * np.random.uniform(-1.0, 1.0)
        if x_new[5] < N3_MIN:
            x_new[5] = N3_MIN
        elif x_new[5] > N3_MAX:
            x_new[5] = N3_MAX

    return x_new


