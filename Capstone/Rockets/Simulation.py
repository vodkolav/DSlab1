
import numpy as np
from scipy.interpolate import Rbf, CubicSpline
from Capstone.Geometry import circumference, normals, magn, cslice, pol2cart, cart2pol, intersections, slerp
from Capstone.Rockets.Harvester import Harvester




class Lagrangian:

    def __init__(self, func,  hull_radius = 4, n = 1000, d = 0.11, window_size=50, interp_method = 'manydumb', ):
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


        #Variables
        self.SimStep=0
        self.I, self.R, self.T, self.X, self.Y = self.grain(func, self.n)

        self.I, self.X, self.Y = self.fill_holes(self.I, self.X, self.Y)

        self.IsNew = np.zeros_like(self.I)
        
        self.A = self.active(self.X, self.Y)

        Hr = np.ones_like(self.T)*self.hr
        self.Hx, self.Hy = pol2cart(Hr, self.T)
        

        self.HSsim = Harvester(["self.SimStep", "C"])

        self.HSintersections = Harvester(varnames=["self.SimStep", "i", "j", "ti", "tj", "Px", "Py",
                                                "clas", "condi", "condj"],
                                        elems='valid')

        self.HScurves = Harvester(varnames=["self.I", "self.X", "self.Y", "self.A", "self.IsNew",
                                    "self.SimStep", "Nx", "Ny", "Ex", "Ey", "filt"], 
                                    on_size_mismatch='error')
                                    #"I1","X1", "Y1", circ, , "isNew" 



    def curve_intersections(self, XY, N):
        """Find intersections within sliding windows.
        For each window start, test the first ray (index i=start) against all following rays in the window.
        Return compact arrays of hits (i, j, ti, tj, Px, Py)."""


        n = XY.shape[0]

        c = XY # segment start points

        c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
        v = c1 - c # segment direction vectors

        filt = np.zeros(n).astype(bool)

        newXY = np.zeros((0,2))
        newI = np.zeros(0)


        for i in range(0, n , 1):
            j = 0 
            c0 = c[i]            # (2,)
            v0 = v[i]            # (2,)

            sl = cslice(i+2, i+self.window_size, n)

            c_block = c[sl] 
            v_block = v[sl] 
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

        return filt, (newI % n).astype(int), newXY[:,0], newXY[:,1]


    # rarefactions: 
    def rarefactions(self, I, X, Y):
        
        XY = np.stack((X,Y), axis=1)
        # n = XY.shape[0]

        c = XY # segment start points

        c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
        v = c1 - c # segment direction vectors    

        m = magn(v[:,0], v[:,1])

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
            print(xy)
        return np.concatenate((xi,yi), axis = 1)


    def fill(self, I, X, Y, divergents):

        XY = np.stack((X,Y), axis=1)
        # n = XY.shape[0]

        c = XY # segment start points

        c1 = np.concatenate((c[-1:,:],c[:-1]), axis=0) # segment end points 
        v = c1 - c # segment direction vectors    

        if self.method == 'dumb':
                # simplest interpolation: just add half the segment vector to the start point of the segment
                newXY = XY[divergents] + v[divergents] * 0.5
                newI = I[divergents]
        else:
            # add multiple points along the segment vector
            # direction vectors of divergents
            # hi = XY[divergents+1,:] - XY[divergents,:]
            hi = v[divergents,:]

            l = np.sqrt(np.sum(hi**2,axis=1))
            jj = (l/self.d).astype(int)       

            dividx = I[divergents]


            if any(l > 1):
                print("oops")

            # cubXY = np.zeros((0,2))
            newXY = np.zeros((0,2))
            newI = np.zeros(0)

            for i,rr in enumerate(dividx):
                j = np.arange(1, jj[i]).reshape(jj[i]-1,1) 

                wat = np.dot(j,hi[[i],:])
                
                xy = XY[rr,:] + wat / jj[i]
                
                # xyl = xy.copy()

                # if (xy > 5).sum() > 0:
                #     print(xyl)

                if self.method == 'interp':
                    slc = cslice(rr-3,rr+3, XY.shape[0])
                    xy = self.interpolate(XY[slc,:], xy)

                if self.method == 'slerp':
                    # not working! 
                    f,t = cslice(rr-1,rr+1, XY.shape[0])
                    xy = slerp(XY[f,:], XY[t,:], self.d)


                newXY = np.concatenate((newXY, xy), axis=0)
                onns = np.ones(jj[i]-1)

                newI = np.concatenate((newI, onns*(rr)), axis=0)
                # print("a", a)
            newI = newI.astype(int)
            newI, newXY

        newX, newY =  newXY[:,0], newXY[:,1]
        return newI, newX, newY


    def active(self, X, Y):
        R,_ = cart2pol(X,Y)
        A = R < self.hr
        return A


    def step(self, I, X, Y, A):
        Nx, Ny = normals(X, Y)

        Ex, Ey =  X + self.d * A * Nx , Y + self.d * A * Ny

        # I = np.asarray(I).ravel()
        if not (X.size == Y.size == Nx.size == Ny.size == I.size):
            raise ValueError('All inputs must have same length')

        # Prepare arrays
        E = np.stack((Ex, Ey), axis=1)  # (n,2)
        N = np.stack((Nx, Ny), axis=1)  # (n,2)


        filt, iI, iX, iY  = self.curve_intersections(E, N) # *(1+s*0.1)
        
        # filt is True where the points should be filtered out / dropped

        self.HScurves.collect(locals())



        isNew1 = np.zeros_like(I)

        X1 = np.insert(Ex, iI, iX, axis=0)
        Y1 = np.insert(Ey, iI, iY, axis=0)

        iF = np.zeros_like(iI).astype(bool)
        F1 = np.insert(filt, iI, iF, axis=0)
        
        news = np.ones_like(iI)
        isNew1 = np.insert(isNew1, iI, news, axis=0)

        if filt.size != 0: #TODO should check if filt has any True instead
            X1 = X1[~F1]
            Y1 = Y1[~F1]
            isNew1 = isNew1[~F1]

        I1 = np.arange(X1.shape[0])

        divergents = self.rarefactions(I1, X1, Y1)

        newI, newX, newY =  self.fill(I1, X1, Y1, divergents )

        news = np.ones_like(newI)



        # TODO: make this a single array operation instead of 3 separate ones
        X1 = np.insert(X1, newI, newX, axis=0)
        Y1 = np.insert(Y1, newI, newY, axis=0)
        I1 = np.arange(X1.shape[0])

        self.IsNew = np.insert(isNew1, newI, news, axis=0)


        A1 = self.active(X1,Y1)

        # S = np.ones_like(A)*s
        C = circumference(X1*A1,Y1*A1)


        return I1, X1, Y1, A1, C



    def grain(self, func, n):
        T = np.linspace(0, np.pi*2, n, endpoint=False) - 0.0001
        # T = np.append(T, T[:1])
        I = np.arange(len(T))
        # Calculate Radius for each Theta
        R = func(T)
        X, Y = pol2cart(R, T)
        return I, R, T, X, Y 


    def fill_holes(self, I, X, Y):
        divergents = self.rarefactions(I, X, Y)

        newI, newX, newY =  self.fill(I, X, Y, divergents )

        X = np.insert(X, newI, newX, axis=0)
        Y = np.insert(Y, newI, newY, axis=0)
        I = np.arange(X.shape[0])
        return I, X, Y


    def run(self, steps = 1):
        """runs the simulation

        Args:
            steps (int, optional): max simulation steps. Defaults to 1.

        Returns:
            dict: dictionary of results data
        """
        
        print('Simulation step size(d):', self.d)
        print('Simulation Steps:', steps)
        print('Curve points (n):', self.n)
        

        for self.SimStep in range(steps):
            print("step:", self.SimStep, " | points:", self.I.shape)

            self.I, self.X, self.Y, self.A, C = self.step(self.I, self.X, self.Y, self.A )

            if sum(self.I.shape) >  self.n * 20 :
                print("too many points, stopping simulation")
                break

            self.HSsim.collect(locals())
            if sum(self.A) == 0:
                print("everything's burnt")
                break

        #TODO return the performance curve



