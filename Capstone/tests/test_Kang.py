
#pip install matplotlib pyfreefem numba pyperclip

import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import os
from copy import deepcopy

from Rockets.Kang.Pipelines import FemSimPipeline
from Rockets.utils import parse_sf_params
from Rockets.Simulation.Plots import preproc_log_data, Log, save_fig

from Benchmarking.Benchmarking import Bench

print(os.getcwd())

simParams = dict(
    nm_o=100, 
    nm_i=100, 
    C_pf=1e8, 
    Ro=1.0, 
    Ri=0.1)

PhysicsParams = dict(
    rho_p = 1700.0,
    c_star = 1400.0,
    a = 2e-4,
    n = 0.3,
    At = 5e-2)

AuxParams = {
    "SimulationClass": "FemSim",
    "steps": 500,
    "R": simParams["Ro"]
}

tracks = {
    "harvest": {
        "curves": {
            "func_name": "dump_curves", 
            "varnames": ["I", "X", "Y", "SimStep", "lvl", "patch", "npoints"],
            "on_size_mismatch": "error"}
            ,
        "mesh": {
            "func_name": "dump_mesh", 
            "varnames": ["mesh_ref.x", "mesh_ref.y", "triangles", "S_ref"],
            "on_size_mismatch": "warn",
            "break_2d_vectors": False}
            ,
        "simulation": {
                "func_name": "dump_simulation", 
                "varnames": ["SimStep", "Web", "Time", 
                             "Perimeter", "npoints", "Thrust", "P_c"],
                "on_size_mismatch": "error"
        }
    }
}

dataroot = "/home/michael/Studies/DSlab/Capstone/data/"

dest = dataroot + "data/Generated-dbg/"

# patt = "*"
# path = Path(dataroot + "SFs")
# paths = list(path.glob(patt +".png"))

def signature(case: dict):
    a = [f"{k}={v}" for k,v in case.items()]
    return " ".join(a)

case1 = dict(
    m = int(4),
    a = 0.7,
    b = 0.7,
    n_1 = 2.585,
    n_2 = 9.281,
    n_3 = 9.281
)

case2 = dict(
    m = int(7),
    a = 0.7,
    b = 0.7,
    n_1 = 2.585,
    n_2 = 1.281,
    n_3 = 1.281
)

case3 = {'m': 5,
 'a': 0.7259468092644616,
 'b': 0.7259468092644616,
 'n_1': 7.223508449351684,
 'n_2': 13.57565243153542,
 'n_3': 13.57565243153542}

paths = [case1, case2]

cases = [{"config":{
              "SFparams": p,
              "simulation": simParams,
              "Physics": PhysicsParams,
              "Aux": AuxParams},
          "ID": 
              {"case_signature": signature(p)},
          "tracks": deepcopy(tracks)
          } for p in paths ]


bench = Bench(benchmarks_root=dataroot + "benchmarks",
              output_root= dataroot + "output",
            #   folder="20260801/1442",
              onerror = "fail")

bench.np_orient = "horz"


bench.configure(FemSimPipeline)


bench.set_cases(cases[:2]) # res) 
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