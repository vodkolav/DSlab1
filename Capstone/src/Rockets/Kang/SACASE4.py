### SACASE4.py  

from Rockets.Kang.Plots import Chamber_Pressure_Plots, Trajectory_Plots

import numpy as np
import matplotlib.pyplot as plt
from Rockets.Kang.saburnbackfinal import solve_burnback, solve_SA, pressure_profile, energy_sf, transition_sf
from time import perf_counter
#import matplotlib2tikz

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


if __name__ == '__main__':
    start = perf_counter()

    # Set target parameters
    m_ref = int(4)
    a_ref = 0.7
    b_ref = a_ref
    n1_ref = 2.585
    n2_ref = 9.281
    n3_ref = n2_ref

    # a_ref = np.random.uniform(A_MIN, A_MAX)

    # if EQUAL_A_B:
    #       b_ref = a_ref
    # else:
    #       b_ref = np.random.uniform(B_MIN, B_MAX)
    # m_ref = np.random.randint(M_MIN, M_MAX)
    # n1_ref = np.random.uniform(N1_MIN, N1_MAX)
    # n2_ref = np.random.uniform(N2_MIN, N2_MAX)
    # # n3_ref = np.random.uniform(N3_MIN, N3_MAX)
    # n3_ref = n2_ref

    x_sf_ref = np.array([a_ref, b_ref, m_ref, n1_ref, n2_ref, n3_ref])
    print("Target Parameters:")
    print(x_sf_ref)

    mesh_ref, S_ref = solve_burnback(a_ref, b_ref, m_ref, n1_ref, n2_ref, n3_ref, plot=True)
    web_ref, Ps_ref, time_ref, perimeters, npoints = pressure_profile(mesh_ref, S_ref, plot=True)

    # Random initial guess parameters
    a0 = np.random.uniform(A_MIN, A_MAX)
    b0 = a0
    m0 = np.random.randint(M_MIN, M_MAX)
    n10 = np.random.uniform(N1_MIN, N1_MAX)
    n20 = np.random.uniform(N2_MIN, N2_MAX)
    n30 = n20

    n10 = n1_ref
    x_sf_0 = np.array([a0, b0, m0, n10, n20, n30])
    print("Random initial parameters:")
    print(x_sf_0)

    mesh0, S0 = solve_burnback(a0, b0, m0, n10, n20, n30, plot=True)
    web_0, Ps_0, time_0, perimeters, npoints = pressure_profile(mesh0, S0, plot=True)

    mse_val = energy_sf(x_sf_0, web_ref, Ps_ref, n_div =100)


    Chamber_Pressure_Plots(web_0, Ps_0, web_ref, Ps_ref, time_0, time_ref, label="Initial")

    energy_sf_curr = lambda x: energy_sf(x, web_ref, Ps_ref, n_div=101)
    transition_sf_curr = lambda x, E: transition_sf(x, E, EQUAL_A_B, EQUAL_N23)

    x_sf_opt, E_pc, list_T, list_x, list_E, best_E, best_X, iteration = solve_SA(
        energy_sf_curr,
        x_sf_0,
        transition_sf_curr,
        T0=10.0,
        T_drop=0.997,
        n_iter=2500,
        tol=0.01,
        verbose=True
    )

    Trajectory_Plots(list_T, list_E)

    x_sf_opt[2] = int(x_sf_opt[2])
    print(f'Optimal parameters = {x_sf_opt}')
    print(f'Reference parameters = {x_sf_ref}')
    print(f'Best parameters = {best_X}')
    print(f'Best error = {best_E}')

    mesh_opt, S_opt = solve_burnback(
        best_X[0],
        best_X[1],
        int(best_X[2]),
        best_X[3],
        best_X[4],
        best_X[5],
        plot=True
    )

    webs, Ps, time_s, perimeters, npoints = pressure_profile(mesh_opt, S_opt, plot=True)

    end = perf_counter()
    code_run_time = end-start
    print(f"Code Run Time: {code_run_time:.4f}")

    Chamber_Pressure_Plots2(web_ref, Ps_ref, webs, Ps, time_ref, time_s, label="Target")

    figure_numbers = plt.get_fignums()
    for i, num in enumerate(figure_numbers):
        fig = plt.figure(num)
        plt.savefig(f"SA_CASE_4_PDFFIG_{i+1}.pdf")

    with open("SA_CASE_4_PARAM.txt", "a") as PARAM:
        PARAM.write(f"{x_sf_ref}\n")
        PARAM.write(f"{x_sf_0}\n")
        PARAM.write(f"{mse_val}\n")
        PARAM.write(f"{best_X}\n")
        PARAM.write(f"{best_E}\n")
        PARAM.write(f"{code_run_time:.4f}\n")
        PARAM.write(f"{iteration:.4f}\n")
