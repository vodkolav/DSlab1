

from Rockets.Simulation.Plots import animate, preproc_curves_data, save_fig
from Rockets.Simulation.Simulation import Lagrangian
# from Capstone.Rockets.Simulation import HScurves, HSintersections, HSsim

from Rockets.Superformula.Formulas import formula1, formula2
from Benchmarking.sensors.Harvester import Harvester
from Benchmarking.telemetry_manager import TelemetryManager
from copy import deepcopy


d = .07
steps = 50
n = 1000
window_size = 70
hull_radius = 4
interp = 'manydumb' #'dumb' #
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


HrvesterConf =  {"varnames": ["self.I", "self.XY", "self.A", "self.IsNew",
                              "self.SimStep", "self.N", "E", "self.d", "filt"],
                 "on_size_mismatch": "error"}
HScurves = Harvester(**HrvesterConf)


func_name = "dump_curves"

func = getattr(SIM, func_name)

func = HScurves.attach_to(func)

setattr(SIM,func_name, func)

SIM.run(steps)
print("Simulation completed.")


HSdata = HScurves.results()
# HSintrsctns = SIM.HSintersections.results()
HSsimdata = SIM.HSsim.results()


curvesdf, colmap =  preproc_curves_data(HSdata)

fig = animate(curvesdf, hull_radius, colmap, width=600, height=600)

save_fig(fig, save_path = "simulation.html")