from pyfreefem import FreeFemRunner
from Rockets.Kang.saburnbackfinal import pressure_profile as pprof
import numpy as np


class FemSim:

    edpscript = "Capstone/src/Rockets/Kang/burnbackpftemplate.edp"


    def __init__(self, SFparams, physics, simParams, **kwargs ):
        """Simulate SRM burnback with finite elements method """
        self.runner: FreeFemRunner = FreeFemRunner(self.edpscript)
        self.SFparams: dict = SFparams
        self.simParams: dict = simParams
        self.physicsParams: dict = physics


    def dump_simulation(self, fnlocals):
        pass


    def dump_curves(self, fnlocals):
        pass



    def run(self, steps = 1):
        """runs the simulation

        Args:
            steps (int, optional): max simulation steps. Defaults to 1.

        Returns:
            dict: dictionary of results data
        """
        mesh, S = self.solve_burnback()

        self.feed_curves(mesh, S)

        web_vals, Pc_MPA, time_vals = self.pressure_profile(mesh, S)

        self.dump_simulation(locals())


    def rename_sf_params(self, params: dict):
        renames = { 'm': 'M',
                    'a': 'A',
                    'b': 'B',
                    'n_1': 'N1',
                    'n_2': 'N2',
                    'n_3': 'N3'}
        return {renames[k]:v for k,v in params.items() if k in renames}


    def solve_burnback(self):

        params = self.rename_sf_params(self.SFparams)
        params.update(self.simParams)

        exports = self.runner.execute(params)

        mesh = exports['Th']
        S = exports['S[]']

        return mesh, S


    def pressure_profile(self, mesh, S, DT=1e-2):
        return pprof(mesh, S, DT=DT, plot = False, **(self.physicsParams))


    def emit(self, From, web_vals):
        for i,lvl in enumerate(web_vals): 
            for j,p in enumerate(From.allsegs[i]):
                p = np.vstack([p,[np.nan, np.nan]])
                idx = np.arange(p.shape[0])
                yield {"I": idx, "X": p[:,0], "Y": p[:,1], 
                    "SimStep": i, "lvl":lvl ,"patch": j , 
                    "npoints": p.shape[0]}


    def feed_curves(self, mesh_ref, S_ref):
        import matplotlib.tri    as mtri
        import matplotlib.pyplot as plt
        plt.ioff()
        triangles = mesh_ref.triangles[:, :3] - 1  # convert to zero-based indices
        tri = mtri.Triangulation(mesh_ref.x, mesh_ref.y, triangles=triangles)

        #TODO: parametrize this too
        DT = 0.01
        n_levels = int(np.max(S_ref)/DT) + 1
        web_vals = np.linspace(0, np.max(S_ref), n_levels)
        
        contours = plt.tricontour(tri, S_ref, levels=web_vals,)

        for data in self.emit(contours, web_vals):
            # I, X, Y, SimStep, lvl, patch, npoints = data.values()
            self.dump_curves(data) 
