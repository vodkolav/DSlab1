### NNDATAGENCASE4.py  

# Coeff and Pc data generation for NN using updated
# burn back analysis and internal ballistics
# Varying Parameters: a = b, m
# Constant Parameters: n1, n2 = n3

import numpy as np
from Rockets.Kang.saburnbackfinal import solve_burnback, pressure_profile

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

PC0 = 101325.0
MPA = 1e6


if __name__ == '__main__':
    pc_buffer = []
    coeff_buffer = []

    for i in range(10000):
        m_ref = np.random.randint(M_MIN, M_MAX)
        a_ref = np.random.uniform(A_MIN, A_MAX)
        b_ref = a_ref
        n1_ref = np.random.uniform(N1_MIN, N1_MAX)
        n2_ref = np.random.uniform(N2_MIN, N2_MAX)
        n3_ref = n2_ref

        x_sf_ref = np.array([a_ref, b_ref, m_ref, n1_ref, n2_ref, n3_ref])
        # print(x_sf_ref)

        mesh_ref, S_ref = solve_burnback(a_ref, b_ref, m_ref, n1_ref, n2_ref, n3_ref, plot=False)
        ts_ref, Ps_ref, time_vals, perimeters, npoints  = pressure_profile( mesh_ref, S_ref, plot=False)

        Pc_vals = np.append(Ps_ref, ts_ref[-1])

        pc_buffer.append(" ".join(map(str, Pc_vals)) + "\n")
        coeff_buffer.append(" ".join(map(str, x_sf_ref) ) + "\n")

        print(i)

    with open("PC_DATA_CASE_4.txt", "a") as PC_VARS,\
         open("COEFF_DATA_CASE_4.txt", "a") as COEFF:
        PC_VARS.writelines(pc_buffer)
        COEFF.writelines(coeff_buffer)
