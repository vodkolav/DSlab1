import matplotlib.pyplot as plt

def Chamber_Pressure_Plots(web, Pc, web_tgt, Ps_tgt, time, time_tgt, label):
    # "Predicted"
    plt.figure()
    plt.plot(web, Pc, label=label, color="blue")
    plt.plot(web_tgt, Ps_tgt, label="Target", color="red", linestyle="--")
    plt.xlabel('Burned Web Thickness [m]')
    plt.ylabel("Chamber Pressure [MPa]")
    # plt.title(f"Chamber Pressure vs Time MSE: {mse_val}")
    plt.legend()
    plt.grid()
    # plt.show()

    plt.figure()
    plt.plot(time, Pc, label="Predicted", color="blue")
    plt.plot(time_tgt, Ps_tgt, label="Target", color="red", linestyle="--")
    plt.xlabel('Time [s]')
    plt.ylabel("Chamber Pressure [MPa]")
    # plt.title(f"Chamber Pressure vs Time MSE: {mse_val}")
    plt.grid()
    plt.legend()
    plt.show()



def Trajectory_Plots(list_T, list_E):
    plt.figure()
    plt.plot(list_T, 'k-', lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Annealing Schedule')

    plt.figure()
    plt.plot(list_E, 'b-', lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Trajectory')
    plt.show()



def Chamber_Pressure_Plots2(web_tgt, Ps_tgt, webs, Ps, time_tgt, time_s, label):
    plt.figure()
    plt.plot(web_tgt, Ps_tgt, 'k-', lw=2, label=label)
    plt.plot(webs, Ps, 'b-', lw=2, label='Result')
    plt.xlabel('Burned Web Thickness [m]')
    plt.ylabel('Chamber Pressure [MPa]')
    # plt.title('Burned Web Thickness vs Chamber Pressure')
    plt.grid()
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(time_tgt, Ps_tgt, 'k-', lw=2, label=label)
    plt.plot(time_s, Ps, 'b-', lw=2, label='Result')
    plt.xlabel('Time [s]')
    plt.ylabel('Chamber Pressure [MPa]')
    # plt.title('Time vs Chamber Pressure')
    plt.grid()
    plt.legend()
    plt.show()