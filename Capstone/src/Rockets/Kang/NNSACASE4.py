### NNSACASE4.py  

import os, tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

from Plots import Chamber_Pressure_Plots, Chamber_Pressure_Plots2, Trajectory_Plots

import numpy as np
import matplotlib.pyplot as plt
from saburnbackfinal import solve_burnback, solve_SA, pressure_profile, energy_sf, transition_sf
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

R_A = 0.1
R_B = 0.1
R_M = 1
R_N1 = 0.25
R_N2 = 0.25
R_N3 = 0.25

EQUAL_A_B = True
EQUAL_N23 = True

PC0 = 101325.0
MPA = 1e6

C_T = 0


if __name__ == '__main__':
    model = nn.Sequential(
    nn.Linear(51, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 4))    

    start = perf_counter()

    model.load_state_dict(torch.load('NN_MODEL_CASE_4.pth'))
    model.eval()

    # INPUT PC or find Pc via coeff
    ###########
    m_in = int(4)
    a_in = 0.7
    b_in = a_in
    n1_in = 2.585
    n2_in = 9.281
    n3_in = n2_in

    # a_in = np.random.uniform(A_MIN, A_MAX)

    # if EQUAL_A_B:
    #       b_in = a_in
    # else:
    #       b_in = np.random.uniform(B_MIN, B_MAX)
    # m_in = np.random.randint(M_MIN, M_MAX)
    # n1_in = np.random.uniform(N1_MIN, N1_MAX)
    # n2_in = np.random.uniform(N2_MIN, N2_MAX)
    # n3_in = n2_in

    x_in = np.array([a_in, b_in, m_in, n1_in, n2_in, n3_in])
    print("Target Parameters")
    print(x_in)

    mesh_in, S_in = solve_burnback(a_in, b_in, m_in, n1_in, n2_in, n3_in, plot=True)
    web_in, Ps_in, time_in = pressure_profile(mesh_in, S_in, plot=True)

    curves_in = []
    t_in = np.linspace(0, web_in[-1], len(Ps_in))
    t_in_interp = np.linspace(0, web_in[-1], 50)
    Pc_in_interp = np.interp(t_in_interp, t_in, Ps_in)
    Pc_NN = np.hstack([Pc_in_interp, web_in[-1]])

    ######################################################################################################

    Pc_input = torch.tensor(Pc_NN.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        coeff_pred = model(Pc_input).squeeze(0).detach().numpy()


    a_pred, m_pred, n1_pred, n2_pred = coeff_pred
    m_pred = int(m_pred)
    b_pred = a_pred
    n3_pred = n2_pred

    x_pred = np.array([a_pred, b_pred, m_pred, n1_pred, n2_pred, n3_pred])
    print("Predicted Parameters")
    print(x_pred)

    mesh_pred, S_pred = solve_burnback(a_pred, b_pred, m_pred, n1_pred, n2_pred, n3_pred, plot=True)
    web_pred, Pc_pred, time_pred = pressure_profile(mesh_pred, S_pred, DT=1e-2, plot=True)

    mse_val = energy_sf(x_pred, web_in, Ps_in, n_div =100)

    Chamber_Pressure_Plots(web_pred, Pc_pred, web_in, Ps_in, time_pred, time_in, label="Predicted")

    energy_sf_curr = lambda x: energy_sf(x, web_in, Ps_in, n_div=101)
    transition_sf_curr = lambda x, E: transition_sf(x, 0, EQUAL_A_B, EQUAL_N23)

    x_sf_opt, E_pc, list_T, list_x, list_E, best_E, best_X, iteration = solve_SA(
        energy_sf_curr,
        x_pred,
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
    print(f'Reference parameters = {x_in}')

    mesh_opt, S_opt = solve_burnback(
        x_sf_opt[0],
        x_sf_opt[1],
        int(x_sf_opt[2]),
        x_sf_opt[3],
        x_sf_opt[4],
        x_sf_opt[5],
        plot=True
    )

    webs, Ps, time_s = pressure_profile(mesh_opt, S_opt, plot=True)

    end = perf_counter()
    code_run_time = end-start
    print(f"Code Run Time: {code_run_time:.4f}")

    Chamber_Pressure_Plots2(web_in, Ps_in, webs, Ps, time_in, time_s, label="Target")

    figure_numbers = plt.get_fignums()
    for i, num in enumerate(figure_numbers):
        fig = plt.figure(num)
        plt.savefig(f"NN_SA_CASE_4_PDFFIG_{i+1}.pdf")

    with open("NN_SA_CASE_4_PARAM.txt", "a") as PARAM:
        PARAM.write(f"{x_in}\n")
        PARAM.write(f"{x_pred}\n")
        PARAM.write(f"{mse_val}\n")
        PARAM.write(f"{best_X}\n")
        PARAM.write(f"{best_E}\n")
        PARAM.write(f"{code_run_time:.4f}\n")
        PARAM.write(f"{iteration:.4f}\n")
