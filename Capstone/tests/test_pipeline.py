import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import os


from Rockets.Simulation.Pipelines import SimPipeline
from Rockets.utils import parse_sf_params

from Benchmarking.Benchmarking import Bench

print(os.getcwd())

simParams = {
"d": .05,
"steps": 70,
"n": 1000,
"window_size": 70,
"hull_radius": 4,
"interp": 'manydumb'
}

dataroot = "/home/michael/Studies/DSlab/Capstone/data/"

dest = dataroot + "data/Generated-dbg/"


patt = "*"
path = Path(dataroot + "SFs")

paths = list(path.glob(patt +".png"))

tracks = {
    "episodes": {
        "sampling_type": "interval_episodes",
        "sampling_value": 1},
    "harvest": {
        "curves": {
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
          "tracks": tracks
          } for p in paths ]


bench = Bench(benchmarks_root=dataroot + "benchmarks",
              output_root= dataroot + "output",
              folder="20260801/1442",
              onerror = "fail")


simppl = SimPipeline()

#case_template = simppl.case_template()
#case_template["tracks"].pop('profile')

bench.configure(simppl)

bench.set_cases(cases[2:7])
# bench.unfurl_grid(case_template, chosengrid)

bench.run_experiments()
