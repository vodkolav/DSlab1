


from numpy import pi, sin, cos, arcsin, arccos, sign, abs, floor, max, deg2rad
from scipy.signal import sawtooth


def trapzoid_signal(t, width=2., slope=1., amp=1., offs=0):
    a = slope*width*sawtooth(2*pi*t/width, width=0.5)/4.
    a[a>amp/2.] = amp/2.
    a[a<-amp/2.] = -amp/2.
    return a + amp/2. + offs

def formula1(funcname:str = "superformula",
                m: int = 3,
                a: float = 1, 
                b: float = 1, 
                n_1: float = 0.7,
                n_2: float = 2,
                n_3: float = 2,
                s: float = 0,
                o: float = 1,
                invert: int = 0
                ):

    """passes parameters

    Args:
        funcname (str, optional): Function name. Defaults to superformula. Choice:[0:0:0].
        m (int, optional): number of lobes. Defaults to 3. Range:[0:10:.1].
        a (float, optional): Amplitude. Defaults to 1. Range:[.5:10:0.5].
        b (float, optional): Another Amplitude. Defaults to 1. Range:[.5:10:0.5].
        n_1 (float, optional): Shape parameter 1. Defaults to -1. Exp:[0.01:1000:.01].
        n_2 (float, optional): Shape parameter 2. Defaults to 2. Exp:[0.01:1000:.01].
        n_3 (float, optional): Shape parameter 3. Defaults to 2. Exp:[0.01:1000:.01].
        s (float, optional): shape rotation in degrees. Defaults to 0. Range:[-180:180:1]
        o (float, optional): vertical transition (offset). Defaults to 1. Range:[0:20:.1]
        invert (int, optional): invert R axis. Defaults to False. Range:[0:2:1]
    """
    # s = 3 # l is the horizontal transition
    # o = 1 # o is the vertical transition (offset)

    s = deg2rad(s)

    selection = {
    "floor"       : lambda phi: floor(m*phi/(2*pi)) + s,  

    "abs"         : lambda phi: abs(m*phi/(2*pi)) + s,  

    "rombus"      : lambda phi: a / (cos(pi*m/4 - ((s + phi) % (2*pi*m/4)))),

    "sea-star"    : lambda phi: a/2 * (cos(m*phi) + 1),

    "risingsun"   : lambda phi: a/2 * (sign(sin(m*phi)) + 1) ,

    "sunflower"   : lambda phi: a*2 * abs(m*phi/(2*pi) - floor(m*phi/(2*pi) + 1/2)) ,
    
    "square-wave" : lambda phi: sin(m*phi) /(10**-a * abs(sin(m*phi))),  

    "trapez-wave" : lambda phi: a   * ((arccos(cos(m*phi+s)) + arcsin(sin(m*phi+s)))/pi) ,

    "trapezoid"   : lambda phi: trapzoid_signal(phi, width=4/m, slope=n_1, amp=a),

    "superformula": lambda phi: (abs(cos(m*phi/4)/a)**n_2 + abs(sin(m*phi/4)/b)**n_3 ) ** (-1/n_1)

    }

    check = [" " not in k for k in selection.keys()]

    assert all(check)

    # mod = 1
    #mod = lambda phi : abs(cos(phi*m))
    #r = mod(r)
    
    if funcname is None:
        # return all available options of the funcname
        return list(selection.keys())

    if funcname not in selection.keys():
        raise ValueError( f"we have no funcname {funcname}. Available functions are: " + str(selection.keys()))

    chosen = selection[funcname]

    def func(phi):

        r = chosen(phi + s)
        if invert:
            r = max(r)-r
        return r + o
    return func



def formula2(funcname:str = "superformula",
                m: int = 3,
                a: float = 1, 
                n_1: float = 0.7,
                n_2: float = 2,
                s: float = 0,
                o: float = 1,
                invert: int = 0
                ):

    """passes parameters

    Args:
        funcname (str, optional): Function name. Defaults to superformula. Choice:[0:0:0].
        m (int, optional): number of lobes. Defaults to 3. Range:[0:20:.1].
        a (float, optional): Amplitude. Defaults to 1. Range:[.5:10:0.5].
        n_1 (float, optional): Shape parameter 1. Defaults to -1. Exp:[0.01:1000:0.01].
        n_2 (float, optional): Shape parameter 2. Defaults to 2. Exp:[0.01:1000:.01].
        s (float, optional): horizontal transition. Defaults to 0. Range:[-360:360:1]
        o (float, optional): vertical transition (offset). Defaults to 1. Range:[-20:20:.1]
        invert (int, optional): invert R axis. Defaults to 0. Range:[0:2:1]
    """

    b = a 
    n_3 = n_2

    return formula1(funcname, m, a, b, n_1, n_2, n_3, s, o, invert)