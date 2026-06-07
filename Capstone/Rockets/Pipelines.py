

import numpy as np
import pandas as pd

from Capstone.Rockets.Plots import dots_and_arrows, Interactive_polar, Shape
from Capstone.Rockets.Simulation import step, run
from Capstone.Superformula.Formulas import formula1, formula2
from Capstone.Geometry import cart2pol, pol2cart, rad2deg, normals
from Capstone.utils import describe, clip, shape, pick

def profile2(theta):
    r = 10 / (np.cos(np.pi/4 - (theta % (np.pi/2))))
    return r

def profile1(theta):
    a = 10 # offset
    b = 5 # amplitude
    c = 7 # number of lobes
    r = a + b * np.cos(c*theta)
    return r


def test():

    from Capstone.Rockets.Simulation import HScurves, HSintersections

    d = .11
    steps = 10
    n = 1000
    args = ('trapez wave', 5, 1, -0.66, 0.95, 0, 1, 1)

    profile = formula2(*args)

    run(profile, d, steps, n)

    HSdata = HScurves.results()
    HSintrsctns = HSintersections.results()

    describe(HSdata)
    describe(HSintrsctns)

    print("Simulation completed.")


if __name__ == "__main__":
    test()

exit()

# One step

# args = ('superformula', 3, 1, 0.7, 2, 0, 1, 0)

# args = ('superformula', 5, 1, -0.38, 0.38, 0, 1, 0)

# args = ('superformula', 5, 1, -0.66, 0.95, 0, 1, 1)

args = ('trapez wave', 5, 1, -0.66, 0.95, 0, 1, 1)

profile = formula2(*args)

d = .11
steps = 20
n = 1000 

T = np.linspace(0, np.pi*2 , n)
I = np.arange(len(T))
# Calculate Radius for each Theta
R = profile(T)

Shape(R, T)

X, Y = pol2cart(R, T)
Nx, Ny = normals(X, Y)
X1, Y1 =  X + d * Nx , Y + d * Ny

# indices = slice(224,324)

indices = slice(None)

# subst = [ii[indices] for ii in [X, Y, Nx, Ny,]]

vrs = "I, X, Y, d, s"

subst = clip(vrs, indices)


describe(subst)

I1, X1, Y1, res, extra =  step(**subst)


tmp = res
tmp.update(extra)
# [print(k, v.shape) for k,v in tmp.items() if type(v) not in (int, float, np.float64)]
#tmp = {k: tmp[k] for k in "I, X, Y, Nx, Ny, X1, Y1, Ex, Ey, Px, Py".split(", ")}
describe(tmp)
#[print(k, v.shape) for k,v in tmp.items()]


dots_and_arrows(**tmp, d = d)


# Full simulation run


d = .11
steps = 40

n = 1000

# Theta (radians). To avoid /div0 start from 1 
T = np.linspace(0, np.pi*2 , n)

R = profile(T)

data, extra = run(profile, d, steps, n)

pick(data[1])


df = [pd.DataFrame(pick(dt)) for dt in data]
df = pd.concat(df, ignore_index=True)
df

# check for nulls 
df.isna().sum()
df[df.Nx.isna() | df.Ny.isna()]


df[['R','T']] = df.apply(lambda r: cart2pol(r['X'], r['Y']), axis=1, result_type='expand')
df['TD']      = df.apply(lambda r: rad2deg(r['T']), axis=1)
df['Step']    = df.apply(lambda r: f"{r.step}",  axis = 1) #  Circumference: {r.circ}",
df['face']    = df.apply(lambda r: f"i={r.I} | {round(r.TD)}°", axis = 1)

Interactive_polar(df)