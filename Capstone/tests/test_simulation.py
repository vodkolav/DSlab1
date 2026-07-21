

from Rockets.Simulation.Plots import animate
from Rockets.Simulation.Simulation import Lagrangian
# from Capstone.Rockets.Simulation import HScurves, HSintersections, HSsim

from Rockets.Superformula.Formulas import formula1, formula2


d = .07
steps = 50
n = 1000
window_size = 70
hull_radius = 4
interp = 'manydumb'
#interp = 'slerp'

trw = {
"funcname":"trapez-wave",
"m":5, "a":1,
"n_1":-0.66, "n_2":0.95,
"s":0, "o":1, "invert":1
}

sfl = {
"funcname":"superformula",
"m":5, "a":1,
"n_1":-0.61, "n_2":0.94,
"s":2, "o":1, "invert":1
}


profile = formula1(**trw)

SIM = Lagrangian(profile, hull_radius, n, d, window_size, interp )


SIM.run(steps)
print("Simulation completed.")


HSdata = SIM.HScurves.results()
HSintrsctns = SIM.HSintersections.results()
HSsimdata = SIM.HSsim.results()

# Hull = {"Hx": SIM.Hx, "Hy": SIM.Hy}

df = animate(SIM, "simulation.html")