

from Rockets.Simulation.Plots import animate, preproc_curves_data, \
                                     Log, preproc_log_data,\
                                     save_fig, make_selector
from Rockets.Simulation.Simulation import Lagrangian
# from Capstone.Rockets.Simulation import HScurves, HSintersections, HSsim

from Rockets.Superformula.Formulas import formula1, formula2
from Benchmarking.sensors.Harvester import Harvester
from Benchmarking.telemetry_manager import TelemetryManager
from Benchmarking.Case import Case
from copy import deepcopy


d = .07
steps = 50
n = 1000
window_size = 70
hull_radius = 4
interp = "interp_parametric" # 'interp' # 'manydumb' #' #'dumb' #
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

path = ".tracks.harvest.curves"



HrvesterConf =  {"varnames": ["self.I", "self.XY", "self.A", "self.IsNew",
                              "self.SimStep", "self.N", "E", "self.d", "filt"],
                 "dump_func": "dump_curves",
                 "on_size_mismatch": "error"}

CAse = Case( {"tracks": {"harvest": {"curves": HrvesterConf}}})

tele = TelemetryManager(CAse)
tele.AttachSensor(SIM, "dump_curves", path)

SIM.tele = tele


SIM.run(steps)
print("Simulation completed.")

tele.collect_sensors(CAse)
tele.collect_log(CAse)

HSdata = CAse.get_path(path)['data']
# HSintrsctns = SIM.HSintersections.results()
HSsimdata = SIM.HSsim.results()

curvesdf, colmap =  preproc_curves_data(HSdata, dopad=True)
fig = animate(curvesdf, hull_radius, colmap, 
              ascending = True, lockscale = True, width=1000, height=600)
save_fig(fig, save_path = "simulation.html")


dummy = {'case_signature': "blabla",
         'case_index': 0}

logdata = CAse.get_path("tracks.log")["data"]

[ld.update(dummy) for ld in logdata]

dflog, colmap = preproc_log_data(logdata)

L = Log(dflog, colmap, hover_data = "message")

save_fig(L, save_path = "Log.html")
