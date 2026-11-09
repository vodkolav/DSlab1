

from Rockets.Simulation.Plots import WebAndPerf, save_fig
from Rockets.Kang.FemSim import FemSim

from Benchmarking.Pipeline import Pipeline
from Benchmarking.telemetry_manager import TelemetryManager

class FemSimPipeline(Pipeline):

    def __init__(self):

        # Order of initializers matters, as some depend on others.
        # If a parameter changes, all downstream initializers must re-run.
        # * since Python 3.7 dicts preserve insertion order
        self.initializers = {
                # ".config.Physics": self.init_physics,
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


    # def init_physics(self, new_case):
    #     pth = "Capstone/src/Rockets/Simulation/assets/Physics.canvas"
    #     self.physics = Physics(pth, overrides=new_case.config["Physics"])


    def init_sim(self, new_case):

        sfparams = new_case.config["SFparams"]
        simparms = new_case.config["simulation"]
        Physics = new_case.config["Physics"]

        self.SIM = FemSim(sfparams, Physics, simparms )

        return self.SIM


    def run_case(self, ):

        self.simResult = self.SIM.run()

        self.tele.print("Fem Simulation completed.")


    def post_case(self):
        # optional.
        # runs after each case is completed, and after telemetry has collected all data for the case.
        # can be used to generate plots, summaries, etc. based on the collected telemetry data.
        pass
        case_file = self.tele.CAse.case_filename(field='case_signature')

        WaP = WebAndPerf(self.tele.CAse)

        save_fig(WaP,save_path= case_file + ".html")


    def results(self):
        return self.simResult

