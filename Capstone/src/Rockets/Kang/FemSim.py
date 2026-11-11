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

    def dump_mesh(self, fnlocals):
        pass

    def run(self, *args, **kwargs):
        """runs the simulation

        Args:
            steps (int, optional): max simulation steps. Defaults to 1.

        Returns:
            dict: dictionary of results data
        """

        params = self.SFparams.copy()
        params.update(self.simParams)
        # params = self.rename_sf_params(params)

        mesh, S = self.solve_burnback(**params)

        self.feed_curves(mesh, S)

        Web, P_c, Time, Perimeter, npoints = self.pressure_profile(mesh, S)
        tpoints = npoints
        Thrust = 0 # Kang does not calculate it. I might later...
        SimStep = np.arange(len(Web))

        self.dump_simulation(locals())


    def solve_burnback(self,
        a, b, m, n_1, n_2, n_3,
        nm_o=100, nm_i=100, C_pf=1e8, Ro=1.0, Ri=0.1):

        params = {
            'M': m,
            'A': a,
            'B': b,
            'N1': n_1,
            'N2': n_2,
            'N3': n_3,
            'NMO': nm_o,
            'NMI': nm_i,
            'CPF': C_pf,
            'RO': Ro,
            'RI': Ri
        }

        exports = self.runner.execute(params)

        mesh = exports['Th']
        S = exports['S[]']

        return mesh, S


    def pressure_profile(self, mesh, S, DT=1e-2):
        return pprof(mesh, S, DT=DT, plot = False, **(self.physicsParams))


    def emit(self, From, web_vals):
        for S,lvl in enumerate(web_vals): 
            for j,p in enumerate(From.allsegs[S]):
                p = np.vstack([p,[np.nan, np.nan]])
                idx = np.arange(p.shape[0])
                idx = idx.max() - idx
                yield {"I": idx, "X": p[:,0], "Y": p[:,1], 
                    "SimStep": S, "lvl":lvl ,"patch": j , 
                    "npoints": p.shape[0]}


    def feed_curves(self, mesh_ref, S_ref):
        import matplotlib.tri    as mtri
        import matplotlib.pyplot as plt
        plt.ioff()
        triangles = mesh_ref.triangles[:, :3] - 1  # convert to zero-based indices
        tri = mtri.Triangulation(mesh_ref.x, mesh_ref.y, triangles=triangles)

        self.dump_mesh(locals())

        #TODO: parametrize this too
        DT = 0.01
        sz = np.max(S_ref) - np.min(S_ref)
        n_levels = int(sz/DT) + 1
        web_vals = np.linspace(np.min(S_ref), np.max(S_ref), n_levels)
        
        contours = plt.tricontour(tri, S_ref, levels=web_vals,)

        for data in self.emit(contours, web_vals):
            # I, X, Y, SimStep, lvl, patch, npoints = data.values()
            self.dump_curves(data) 
