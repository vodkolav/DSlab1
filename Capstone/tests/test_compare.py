import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from copy import deepcopy

from Rockets.Pipelines.CompareSims import CompareSimsPipeline
from Rockets.Simulation.Pipelines import SimPipeline

from Rockets.Simulation.Plots import preproc_log_data, Log, save_fig
import SFparams 
from Benchmarking.Benchmarking import Bench
from Benchmarking.utils import JQ

print(os.getcwd())

simParams = {"Lagrangian": 
                {
                "Class": "Lagrangian",
                "steps": 500,
                "n": 800,
                "window_size": 100, 
                "interp":  "interp_parametric" #'manydumb'
                },
             "FemSim": dict(
                Class="FemSim",
                nm_o=100, 
                nm_i=100, 
                C_pf=1e8, 
                Ro=1.0, 
                Ri=0.1)
            }


PhysicsParams = {"Lagrangian": {
                    'L': 1,
                    'R': 1,
                    'd_{t}': 0.05,
                    'd_{e}': 0.1505,
                    'a': 1.24e-5,
                    'n': 0.4,
                    'rho_{p}': 1600,
                    'C_{v}': 1500,
                    'C_{FO}': 1.6,
                    'gamma': 1.2,
                    'P_{a}': 101325,
                    'R_{P}': 1/70,
                },
                "FemSim": dict(
                    At = 5e-2,
                    a = 2e-4,
                    n = 0.3,
                    rho_p = 1700.0,
                    c_star = 1400.0)
        }

# TODO: maybe better to establish some kind of reference here
AuxParams = {
    "steps": 500,
    "R": 1.0
}

tracks = {
    "Lagrangian": {
        "harvest": {
            "curves": {
                "func_name": "dump_curves", 
                "varnames": ["self.I", "self.XY", "self.A", "self.B", "F",
                            "self.SimStep", "self.N", "E", "self.d"],
                "on_size_mismatch": "error"},
            "simulation": {
                    "func_name": "dump_simulation", 
                    "varnames": ["self.SimStep","self.Web", "self.Time",
                                "Perimeter", "npoints", "tpoints", "Thrust", "P_c"],
                    "on_size_mismatch": "error"
            }
        }
    },
    "FemSim": {
        "harvest": {
            "curves": {
                "func_name": "dump_curves", 
                "varnames": ["I", "X", "Y", "SimStep", "lvl", "patch", "npoints"],
                "on_size_mismatch": "error"},
            "mesh": {
                "func_name": "dump_mesh", 
                "varnames": ["mesh_ref.x", "mesh_ref.y", "triangles", "S_ref"],
                "on_size_mismatch": "warn",
                "break_2d_vectors": False},
            "simulation": {
                    "func_name": "dump_simulation", 
                    "varnames": ["SimStep", "Web", "Time", 
                                "Perimeter", "npoints", "tpoints", "Thrust", "P_c"],
                    "on_size_mismatch": "error"
                }
            }
        }
    }

dataroot = "/home/michael/Studies/DSlab/Capstone/data/"

dest = dataroot + "data/Generated-dbg/"


def signature(sf: dict, sim:dict):

    fields = ["Class"]
    a = [f"{k}={v}" for k,v in sim.items() if k in fields]
    
    a.extend([f"{k}={v}" for k,v in sf.items()])

    # a.extend([f"{k}={v}" for k,v in sim.items()])
    return " ".join(a)


# files = SFparams.from_screenshots(dataroot + "SFs")
files = SFparams.from_consts()

sims = ["Lagrangian", "FemSim"]

cases = []

for s in sims:
    cases.extend([
        {"config":{
              "SFparams": p["SFparams"],
              "simulation": deepcopy(simParams[s]),
              "Physics": PhysicsParams[s],
              "Aux": AuxParams},
          "ID":{ 
              "file": p["file"],
              "case_signature": signature(p["SFparams"], simParams[s])},
          "tracks": deepcopy(tracks[s])
          } for p in files ])



bench = Bench(benchmarks_root=dataroot + "benchmarks",
              output_root= dataroot + "output",
            #    folder="20261110/1737",
              onerror = "fail")

bench.np_orient = "horz"


#case_template = simppl.case_template()
#case_template["tracks"].pop('profile')

bench.configure(CompareSimsPipeline)


bench.set_cases(cases) # res) 
# bench.unfurl_grid(case_template, chosengrid)

bench.run_experiments()

Experiment_Log = bench.TrackWithID('log')
    
logdf, log_cmap = preproc_log_data(Experiment_Log)


before = logdf.shape
logdf = logdf.drop_duplicates('time', keep = 'last')
if logdf.shape != before:
    bench.TELE.warning("Experiment Log has duplicate records", "shape before:", before, "shape after:", logdf.shape)


L = Log(logdf, log_cmap, hover_data = ['message','traceback'],  render_mode = 'SVG')

save_fig(L, save_path = bench.output_folder +  "/Log.html")