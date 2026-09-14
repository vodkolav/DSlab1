import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import os
from copy import deepcopy

from Rockets.Simulation.Pipelines import SimPipeline
from Rockets.utils import parse_sf_params
from Rockets.Simulation.Plots import preproc_log_data, Log, save_fig

from Benchmarking.Benchmarking import Bench

print(os.getcwd())

simParams = {
"d": .05,
"steps": 70,
"n": 1000,
"window_size": 70,
"hull_radius": 4,
"interp":  "interp_parametric" #'manydumb'
}

dataroot = "/home/michael/Studies/DSlab/Capstone/data/"

dest = dataroot + "data/Generated-dbg/"


patt = "*"
path = Path(dataroot + "SFs")

paths = list(path.glob(patt +".png"))

tracks = {
    "episodes": {
        "func_name": "step", 
        "summary_func": "step_summary",
        "sampling_type": "interval_episodes",
        "sampling_value": 1},
    "harvest": {
        "curves": {
            "func_name": "dump_curves", 
            "varnames": ["self.I", "self.XY", "self.A", "self.IsNew",
                         "self.SimStep", "self.N", "E", "self.d", "filt"],
            "on_size_mismatch": "error"}
            }
}


cases = [{"config":{
              "file":
                  {"path": p.resolve().as_posix() , 
                   "name": p.name, 
                   "stem": p.stem}, 
              "SFparams": parse_sf_params(p.stem),
              "simulation": simParams,},
          "ID": 
              {"case_signature": p.stem},
          "tracks": deepcopy(tracks)
          } for p in paths ]


bench = Bench(benchmarks_root=dataroot + "benchmarks",
              output_root= dataroot + "output",
              folder="20260801/1442",
              onerror = "fail")


simppl = SimPipeline

#case_template = simppl.case_template()
#case_template["tracks"].pop('profile')

bench.configure(simppl)

bench.set_cases(cases) #[2:4]) 
# bench.unfurl_grid(case_template, chosengrid)

bench.run_experiments()

Experiment_Log = bench.TrackWithID('log')
    
logdf, log_cmap = preproc_log_data(Experiment_Log)


before = logdf.shape
logdf = logdf.drop_duplicates('time', keep = 'last')
if logdf.shape != before:
    bench.TELE.warning("Experiment Log has duplicate records", "shape before:", before, "shape after:", logdf.shape)


L = Log(logdf, log_cmap, hover_data = 'message',  render_mode = 'SVG')

save_fig(L, save_path = bench.output_folder +  "/Log.html")