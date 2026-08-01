
import numpy as np
from scipy.interpolate import Rbf, CubicSpline
from Rockets.Geometry import circumference, normals, Magn, cslice, pol2cart, cart2pol, intersections, slerp
from Benchmarking.sensors.Harvester import Harvester

from Benchmarking.telemetry_manager import TelemetryManager



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

        self.tele = TelemetryManager()
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
        
        # segment direction vectors; size: [n,2]
        self.S = []

        #Variables
        self.SimStep=0
        self.I, self.RT, self.XY = self.grain(func, self.n)

        # XY = np.stack((self.X,self.Y), axis=1)
        self.segments(self.XY)
        self.I, self.XY = self.fill_holes(self.I, self.XY)

        self.IsNew = np.zeros_like(self.I)
        
        self.A = self.active(self.XY)

        # Hr = np.ones_like(self.T)*self.hr
        # self.Hx, self.Hy = pol2cart(Hr, self.T)

        self.HSsim = Harvester(["self.SimStep", "C"])

        self.HSintersections = Harvester(varnames=["self.SimStep", "i", "j", "ti", "tj", "Px", "Py",
                                                "clas", "condi", "condj"],
                                        elems='valid')

        # self.HScurves = Harvester(varnames=["self.I", "self.XY", "self.A", "self.IsNew",
        #                             "self.SimStep", "self.N", "E", "self.d", "filt"], 
        #                             on_size_mismatch='error')
                                    #"I1","X1", "Y1", circ, , "isNew" 


    def dump_curves(self, fnlocals):
        pass

    def segments(self, XY):
        Ends = np.concatenate((XY[-1:,:],XY[:-1,:]), axis=0) # segment end points 
        self.S = Ends - XY # segment direction vectors


    def curve_intersections(self, XY):
        """Find intersections within sliding windows.
        For each window start, test the first ray (index i=start) against all following rays in the window.
        Return compact arrays of hits (i, j, ti, tj, Px, Py)."""


        n = XY.shape[0]

        # XY = XY # segment start points


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
            Px = P[:,0]
            Py = P[:,1]

            if any(condi & condj):
                j_block = np.min(np.where(condi & condj))
                j = i + 2 + j_block
                sl = cslice(i,j,n)
                filt[sl] = True


                newXY = np.concatenate((newXY, P[j_block:j_block+1,:]), axis=0)
                newI = np.concatenate((newI, [j]), axis=0)


            self.HSintersections.collect(locals())

        return filt, (newI % n).astype(int), newXY


    # rarefactions: 
    def rarefactions(self):
        
        # XY = np.stack((X,Y), axis=1)
        # # n = XY.shape[0]

        # c = XY # segment start points

        # c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
        # v = c1 - c # segment direction vectors    

        v = self.S

        m = Magn(v)

        # isNew = np.zeros_like(X, dtype=bool)

        quantiles = np.sum(m[:, None] > m, axis=1) / (len(m) - 1)
        # only take the points in the top 2% of segment lengths, e.g points that diverged the most
        divergents = (m > self.d) & (quantiles > .95) 

        return divergents


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

        #XY = np.stack((X,Y), axis=1)
        # n = XY.shape[0]

        # c = XY # segment start points

        # c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
        # v = c1 - c # segment direction vectors    
        # v = self.S

        if self.method == 'dumb':
                # simplest interpolation: just add half the segment vector to the start point of the segment
                newXY = XY1[divergents] + self.S[divergents] * 0.5
                newI = I[divergents]
        else:
            # add multiple points along the segment vector
            # direction vectors of divergents
            # hi = XY[divergents+1,:] - XY[divergents,:]
            hi = self.S[divergents,:]

            l = Magn(hi)
            jj = (l/self.d).astype(int)       

            dividx = I[divergents]


            if any(l > 1):
                self.tele.print("oops")

            # cubXY = np.zeros((0,2))
            newXY = np.zeros((0,2))
            newI = np.zeros(0)

            for i,rr in enumerate(dividx):

                xy = []
                onns = np.ones(jj[i]-1)

                if self.method in ('manydumb', 'interp'):
                    j = np.arange(1, jj[i]).reshape(jj[i]-1,1) 

                    wat = np.dot(j,hi[[i],:])
                    
                    xy = XY1[rr,:] + wat / jj[i]
                    
                    # xyl = xy.copy()

                    # if (xy > 5).sum() > 0:
                    #     print(xyl)

                if self.method == 'interp':
                    slc = cslice(rr-3,rr+3, XY1.shape[0])
                    xy = self.interpolate(XY1[slc,:], xy)

                if self.method == 'slerp':
                    # not working! 
                    f,t = cslice(rr-1,rr+1, XY1.shape[0])
                    newN = slerp(self.N[f,:], self.N[t,:], self.d)
                    xy = self.advance(XY1[f,:], newN, self.d, True)
                    onns = np.ones(xy.shape[0])

                newXY = np.concatenate((newXY, xy), axis=0)

                newI = np.concatenate((newI, onns*(rr)), axis=0)
                # print("a", a)
            newI = newI.astype(int)
            newI, newXY

        # newX, newY =  newXY[:,0], newXY[:,1]
        return newI, newXY


    def active(self, XY):
        R = Magn(XY)
        A = R < self.hr
        return A


    def advance(self, O, N, d, A):
        dd = np.ones([2,1]) * d
        # Ex, Ey =  X + self.d * A * Nx , Y + self.d * A * Ny
        E =  O + (dd * A).T * N
        return E


    def step(self, I, XY, A):

        # XY = np.stack((X,Y), axis=1)

        self.N = normals(XY)

        E = self.advance(XY, self.N, self.d, A)
        if not (XY.shape[0]  == self.N.shape[0] == I.shape[0]):
            raise ValueError('All inputs must have same length')

        # Prepare arrays
        # E = np.stack((Ex, Ey), axis=1)  # (n,2)

        self.segments(E)

        filt, iI, iXY  = self.curve_intersections(E) # *(1+s*0.1)


        # filt is True where the points should be filtered out / dropped

        # self.HScurves.collect(locals())
        self.dump_curves(locals())



        isNew1 = np.zeros_like(I)

        XY1 = np.insert(E, iI, iXY, axis=0)
        # Y1 = np.insert(Ey, iI, iY, axis=0)

        iF = np.zeros_like(iI).astype(bool)
        F1 = np.insert(filt, iI, iF, axis=0)
        
        news = np.ones_like(iI)
        isNew1 = np.insert(isNew1, iI, news, axis=0)

        if filt.size != 0: #TODO should check if filt has any True instead
            XY1 = XY1[~F1]
            # Y1 = Y1[~F1]
            isNew1 = isNew1[~F1]

        I1 = np.arange(XY1.shape[0])

        self.segments(XY1)

        divergents = self.rarefactions()

        newI, newXY =  self.fill(I1, XY1, divergents )

        news = np.ones_like(newI)


        XY1 = np.insert(XY1, newI, newXY, axis=0)
        I1 = np.arange(XY1.shape[0])

        self.IsNew = np.insert(isNew1, newI, news, axis=0)


        A1 = self.active(XY1)
        dd = np.ones([2,1]) * self.d
        dA1 = (dd * A1).T
        # S = np.ones_like(A)*s
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


    def fill_holes(self, I, XY):
        divergents = self.rarefactions()

        newI, newXY =  self.fill(I, XY, divergents )

        XY = np.insert(XY, newI, newXY, axis=0)
        # Y = np.insert(Y, newI, newY, axis=0)
        I = np.arange(XY.shape[0])
        return I, XY


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
        self.HSsim.collect(locals())

        for self.SimStep in range(steps):
            self.tele.ping("step:", self.SimStep, " | points:", self.I.shape)

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



