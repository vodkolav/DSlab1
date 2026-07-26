

from Rockets.Simulation.Plots import dots_and_arrows, Interactive_polar, Shape, WebAndPerf
from Rockets.Simulation.Simulation import Lagrangian
from Rockets.Superformula.Formulas import formula1, formula2
from Rockets.utils import describe, clip, shape, pick

from Benchmarking.Pipeline import Pipeline
from Benchmarking.telemetry_manager import TelemetryManager

class SimPipeline(Pipeline):

    def __init__(self):
        

        # Order of initializers matters, as some depend on others.
        # If a parameter changes, all downstream initializers must re-run.
        # * since Python 3.7 dicts preserve insertion order
        self.initializers = {
                ".SFparams": self.init_sim,
            }
        self.current_case = {} # TODO: make it a property in base class? 


    @property
    def case_file(self):
        return self.tele.case_filename()


    def set_telemetry(self, tele: TelemetryManager):
        #TODO: check if tele can just be passed to __init__ on creation of pipeline
        # or do we require a separate set_telemetry step? 

        self.tele = tele
        
        # pth = ".tracks.resources"
        # conf = get_path(pth, self.case_template())
        # at this point case is not yet initialized, so we use default config from case_template
        # self.tele.AttachSensor(self, "execute", pth, config = conf)


    def init_telemetry(self):
        # re-runs for every new case
        # TODO: attach monitors for particular pipeline components 

        tracks = [
            ".tracks.episodes",
            ".tracks.harvest.curves"
            #,".tracks.resources" 
            #,".tracks.log",
            #,".model_tts.tracks.log"
            ] 

        self.tele.AttachSensor(self.SIM, "step", tracks[0], summary_func = "step_summary")
        self.tele.AttachSensor(self.SIM, "dump_curves", tracks[1], summary_func = "step_summary")

        self.SIM.tele = self.tele

    def init_sim(self, new_case):

        self.sfparams   = new_case["SFparams"]
        self.simparams  = new_case["simulation"] 
        self.fileparams = new_case["file"]

        self.profile = formula1(**self.sfparams)

        self.SIM = Lagrangian(self.profile, **self.simparams)



    def run_case(self, ): #Case, dest
 
        self.SIM.run(self.simparams["steps"])

        self.tele.print("Simulation completed.")

        
        # HSintrsctns = SIM.HSintersections.results()
        # HSsimdata = SIM.HSsim.results()

        # self.current_case['file']['stem']
        # WebAndPerf(self.SIM, self.case_file + ".html")


    def results(self):
        return self.tele.results()

