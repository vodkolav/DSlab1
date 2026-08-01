
from Rockets.Simulation.Plots import preproc_curves_data, preproc_sim_data
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

        # when this func runs, the self.tele.CAse is not yet updated to the new case, 
        # so we can't use it to get the config for the new case.
        # we have to use the new_case argument instead, which is passed to this func by the pipeline.

        self.profile = formula1(**new_case.config["SFparams"])

        self.SIM = Lagrangian(self.profile, **new_case.config["simulation"])


    def run_case(self, ):
        self.simResult = self.SIM.run(self.case_config["simulation"]["steps"])

        self.tele.print("Simulation completed.")


    def post_case(self, tracks):
        # optional.
        # runs after each case is completed, and after telemetry has collected all data for the case.
        # can be used to generate plots, summaries, etc. based on the collected telemetry data.

        simdf = preproc_sim_data(tracks['episodes']['data'])
        curvesdf =  preproc_curves_data(tracks['harvest']['curves']['data'])
        case_file = self.tele.case_filename()

        WebAndPerf(simDF=simdf, 
                   curvesDF=curvesdf, 
                   hull_radius= self.simparams["hull_radius"], 
                   save_path= case_file + ".html")


    def results(self):
        return self.simResult

