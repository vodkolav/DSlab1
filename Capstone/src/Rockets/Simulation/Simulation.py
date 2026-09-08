
import numpy as np
from scipy.interpolate import Rbf, CubicSpline
from Rockets.Geometry import circumference, normals, Magn, cslice, pol2cart,\
                            cart2pol, intersections, slerp, rarefactions, segments
from Benchmarking.sensors.Harvester import Harvester

from Benchmarking.telemetry_manager import DummyTelemetryManager



class Lagrangian:

    def __init__(self, func,  hull_radius = 4, n = 1000, d = 0.11, window_size=50, interp_method = 'manydumb', **kwargs ):
        """Simulate SRM burnback

        Args:
            func (def): function defining the curve. must be of form Radius = f(Theta)
            hull_radius (int, optional): _description_. Defaults to 4.
            n (int, optional): simulation points. Defaults to 1000.
            d (float, optional): step size. Defaults to .11.
            window_size (int, optional): window size for intersections detection. Defaults to 50.
            interp_method (str, optional): interpolation method for filling of rearefactions. Defaults to 'manydumb'.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        self.tele = DummyTelemetryManager()
        #Settings
        if window_size < 2:
            raise ValueError('window_size must be >= 2')
        self.window_size = window_size

        self.d = d

        avail_methods = ['dumb', 'manydumb', 'interp', 'slerp']
        if interp_method not in avail_methods:
            raise ValueError("method must be one of " + str(avail_methods))
        self.method = interp_method

        self.hr = hull_radius
        self.func = func
        self.n = n
        # TODO: validations of func and casing size


        # Caching variables: only relevant to current step.
        
        # normals; size: [n,2]
        self.N = []


        #Variables
        self.SimStep=0
        self.I, self.RT, self.XY = self.grain(func, self.n)

        # segment direction vectors; size: [n,2]
        self.S = segments(self.XY)

        self.M = Magn(self.S)

        divergents = rarefactions(self.M, self.d)

        newI, newXY =  self.fill(self.I, self.XY, divergents )

        XY = np.insert(self.XY, newI, newXY, axis=0)
        I = np.arange(self.XY.shape[0])
        self.I, self.XY = I, XY

        self.IsNew = np.zeros_like(self.I)
        
        self.A = self.active(self.XY)

        self.HSsim = Harvester(["self.SimStep", "C", "npoints"])



    def dump_curves(self, fnlocals):
        pass


    def dump_intersections(self, fnlocals):
        pass


    def curve_intersections(self, XY):
        """Find intersections within sliding windows.
        For each window start, test the first ray (index i=start) against all following rays in the window.
        Return compact arrays of hits (i, j, ti, tj, Px, Py)."""


        n = XY.shape[0]


        filt = np.zeros(n).astype(bool)
        newXY = np.zeros((0,2))
        newI = np.zeros(0)


        for i in range(0, n , 1):
            j = 0 
            c0 = XY[i]            # segment origin
            v0 = self.S[i]        # segment direction

            sl = cslice(i+2, i+self.window_size, n)

            c_block = XY[sl] # segments origin
            v_block = self.S[sl] # segments direction
            # rest of the vectors in window (shifted by 2 to avoid adjacent segments)
            # Adjacent segments by definition intersect at their shared vertex, which is not a valid intersection for our purposes. 
            # By shifting by 2, we ensure that we are only checking for intersections between non-adjacent segments.


            ti, tj, valid = intersections(c0, v0, c_block, v_block) 
            # valid = valid & (((l < ti) & (ti  < u)) | ((l < tj ) & (tj < u)))

            condi = (0 <= ti) & (ti <= 1)
            condj = (0 <= tj) & (tj <= 1)

            clas = valid*1 + condi*2 + condj*3

            valid = valid & condi & condj

            P = c0 + np.stack((ti,ti), axis=1) * v0
            Px = P[:,0] #TODO: harvester can break these up by himself
            Py = P[:,1]

            if any(condi & condj):
                j_block = np.min(np.where(condi & condj))
                j = i + 2 + j_block
                sl = cslice(i,j,n)
                filt[sl] = True


                newXY = np.concatenate((newXY, P[j_block:j_block+1,:]), axis=0)
                newI = np.concatenate((newI, [j]), axis=0)

            self.dump_intersections(locals())

        return filt, (newI % n).astype(int), newXY




    def interpolate(self, XY, xy):
        x,y = np.split(XY,2, axis=1)
        xi,_ = np.split(xy,2, axis=1)
        # TODO: option to choose interpolation method in simulation settings.
        # rbf = Rbf(x, y)
        rbf = Rbf(x.squeeze(), y.squeeze())
        yi = rbf(xi)

        if (np.abs(yi) > 5).sum() > 0:
            self.tele.print(xy)
        return np.concatenate((xi,yi), axis = 1)


    def fill(self, I, XY1, divergents):


        if self.method == 'dumb':
                # simplest interpolation: just add half the segment vector to the start point of the segment
                newXY = XY1[divergents] + self.S[divergents] * 0.5
                newI = I[divergents]
        else:
            # add multiple points along the segment vector
            
            I_d = I[divergents] # indices of the divergent segments

            S_d = self.S[divergents,:,None] # segments direction vectors of divergents
                                            # The 'None' pre-expands S into 3rd dimension 
                                            # so that s elements play nicely with np.dot down the line

            l_d = self.M[divergents]  # lengths of those segments

            P_d = (l_d/self.d).astype(int) # number of points to add along each such segment


            newXY = np.zeros((0,2))
            newI = np.zeros(0)

            for i,s,p in zip(I_d, S_d, P_d):

                xy = []
                onns = np.ones(p)

                if self.method in ('manydumb', 'interp'):
                    ofsts = np.linspace([0,],[1,],p+2)[1:-1]

                    wat = np.dot(ofsts, s.T)

                    j = cslice(i,i+1,XY1.shape[0])
                    xy  = XY1[j,:] + wat 


                if self.method == 'interp':
                    slc = cslice(i-3,i+3, XY1.shape[0])
                    xy = self.interpolate(XY1[slc,:], xy)

                if self.method == 'slerp':
                    # not working! 
                    f,t = cslice(i-1,i+1, XY1.shape[0])
                    newN = slerp(self.N[f,:], self.N[t,:], self.d)
                    xy = self.advance(XY1[f,:], newN, self.d, True)
                    onns = np.ones(xy.shape[0])

                newXY = np.concatenate((newXY, xy), axis=0)

                newI = np.concatenate((newI, onns*(i)), axis=0)

            newI = newI.astype(int)
            newI, newXY

        return newI, newXY


    def active(self, XY):
        R = Magn(XY)
        A = R < self.hr
        return A


    def advance(self, O, N, d, A):
        dd = np.ones([2,1]) * d
        E =  O + (dd * A).T * N
        return E


    def step(self, I, XY, A):

        self.N = normals(XY)

        E = self.advance(XY, self.N, self.d, A)
        if not (XY.shape[0]  == self.N.shape[0] == I.shape[0]):
            raise ValueError('All inputs must have same length')


        self.S = segments(E)

        filt, iI, iXY  = self.curve_intersections(E) # *(1+s*0.1)
        # filt is True where the points should be filtered out / dropped

        self.dump_curves(locals())

        # from this point on the old I, XY, A are obsolete

        isNew1 = np.zeros_like(I)

        XY1 = np.insert(E, iI, iXY, axis=0)

        iF = np.zeros_like(iI).astype(bool)
        F1 = np.insert(filt, iI, iF, axis=0)
        
        news = np.ones_like(iI)
        isNew1 = np.insert(isNew1, iI, news, axis=0)

        if filt.size != 0: #TODO should check if filt has any True instead
            XY1 = XY1[~F1]
            isNew1 = isNew1[~F1]

        I1 = np.arange(XY1.shape[0])

        self.S = segments(XY1)

        self.M = Magn(self.S)

        divergents = rarefactions(self.M, self.d)

        newI, newXY =  self.fill(I1, XY1, divergents )

        news = np.ones_like(newI)

        XY1 = np.insert(XY1, newI, newXY, axis=0)
        I1 = np.arange(XY1.shape[0])

        self.IsNew = np.insert(isNew1, newI, news, axis=0)


        A1 = self.active(XY1)
        dd = np.ones([2,1]) * self.d
        dA1 = (dd * A1).T
        C = circumference(XY1*dA1)

        return I1, XY1, A1, C


    def step_summary(self, mode = "data"):
        if mode == "index":
            return self.SimStep
        else:
            return self.HSsim.last_step()

    def grain(self, func, n):
        T = np.linspace(0, np.pi*2, n, endpoint=False) - 0.0001
        # T = np.append(T, T[:1])
        I = np.arange(len(T))
        # Calculate Radius for each Theta
        R = func(T)
        RT = np.stack((R,T), axis=1)

        X, Y = pol2cart(R, T)
        XY = np.stack((X,Y), axis=1)

        return I, RT, XY 


    def run(self, steps = 1):
        """runs the simulation

        Args:
            steps (int, optional): max simulation steps. Defaults to 1.

        Returns:
            dict: dictionary of results data
        """
        
        self.tele.print('Simulation step size(d):', self.d)
        self.tele.print('Simulation Steps:', steps)
        self.tele.print('Curve points (n):', self.n)

        C = 0 
        npoints = len(self.I)
        self.HSsim.collect(locals())

        for self.SimStep in range(steps):
            npoints = len(self.I)
            self.tele.ping("step:", self.SimStep, " | points:", npoints, " | circumference:", C)

            self.I, self.XY, self.A, C = self.step(self.I, self.XY, self.A )

            if sum(self.I.shape) >  self.n * 20 :
                self.tele.error("too many points, stopping simulation")
                break

            self.HSsim.collect(locals())
            if sum(self.A) == 0:
                self.tele.print("everything's burnt")
                break

        #TODO return the performance curve. think more on representation
        return self.HSsim.results()

