

from Rockets.Simulation.Plots import WebAndPerf, save_fig
from Rockets.Simulation.Simulation import Lagrangian, Physics
from Rockets.Kang.FemSim import FemSim
from Rockets.Superformula.Formulas import formula1, formula2
from Rockets.utils import describe, clip, shape, pick, trypop

from Benchmarking.Pipeline import Pipeline
from Benchmarking.telemetry_manager import TelemetryManager
from Benchmarking.Case import Case
from copy import deepcopy

class CompareSimsPipeline(Pipeline):

    def __init__(self):

        # Order of initializers matters, as some depend on others.
        # If a parameter changes, all downstream initializers must re-run.
        # * since Python 3.7 dicts preserve insertion order
        self.initializers = {
                ".config.Physics": self.init_physics,
                ".config.SFparams": self.init_sim,
            }
        self.simResult = []

    @property
    def case_config(self):
        return self.tele.CAse.config # TODO: make it a property in base class? 


    @property
    def case_file(self):
        raise NotImplemented
        return 


    def set_telemetry(self, tele: TelemetryManager):
        #TODO: check if tele can just be passed to __init__ on creation of pipeline
        # or do we require a separate set_telemetry step? 

        self.tele = tele


    def init_physics(self, new_case: Case):
        simparms = new_case.config["simulation"]
        self.SimClass = trypop(simparms, "Class")

        if self.SimClass == "Lagrangian":
            pth = "Capstone/src/Rockets/Simulation/assets/Physics.canvas"
            self.physics = Physics(pth, overrides=new_case.config["Physics"])

        elif self.SimClass == "FemSim":
            self.physics = new_case.config["Physics"]

        else:
            raise ValueError(f"We don't know what is '{self.SimClass}' simulation class")



    def init_sim(self, new_case: Case):
        sfparams = new_case.config["SFparams"]
        simparms = new_case.config["simulation"]


        self.SimClass = trypop(simparms, "Class")

        if self.SimClass == "Lagrangian":

            self.profile = formula1(**sfparams)
            self.SIM = Lagrangian(self.profile, self.physics, **simparms)

        elif self.SimClass == "FemSim":
            Physics = new_case.config["Physics"]
            self.SIM = FemSim(sfparams, Physics, simparms )

        else:
            raise ValueError(f"We don't know what is '{self.SimClass}' simulation class")

        return self.SIM


    def run_case(self, ):
        self.simResult = self.SIM.run(self.case_config["Aux"]["steps"])

        self.tele.print(f"{self.SimClass} Simulation completed.")


    def post_case(self):
        # optional.
        # runs after each case is completed, and after telemetry has collected all data for the case.
        # can be used to generate plots, summaries, etc. based on the collected telemetry data.

        case_file = self.tele.CAse.case_filename(field='case_signature')

        WaP = WebAndPerf(self.tele.CAse)

        save_fig(WaP,save_path= case_file + ".html")


    def results(self):
        return self.simResult

