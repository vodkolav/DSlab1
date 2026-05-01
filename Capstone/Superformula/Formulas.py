


from numpy import pi, sin, cos, arcsin, arccos, sign, abs, floor



def formula1(funcname:str = "superformula",
                m: int = 3,
                a: float = 1, 
                b: float = 1, 
                n_1: float = 0.7,
                n_2: float = 2,
                n_3: float = 2,
                s: float = 0,
                o: float = 1
                ):

    """passes parameters

    Args:
        funcname (str, optional): Function name. Defaults to superformula. Choice:[0:0:0].
        m (int, optional): number of lobes. Defaults to 3. Range:[0:10:.1].
        a (float, optional): Amplitude. Defaults to 1. Range:[.5:10:0.5].
        b (float, optional): Another Amplitude. Defaults to 1. Range:[.5:10:0.5].
        n_1 (float, optional): Shape parameter 1. Defaults to -1. Dec:[-1:3:0.01].
        n_2 (float, optional): Shape parameter 2. Defaults to 2. Dec:[-1:3:.01].
        n_3 (float, optional): Shape parameter 3. Defaults to 2. Dec:[-1:3:.01].
        s (float, optional): horizontal transition. Defaults to 0. Range:[-360:360:1]
        o (float, optional): vertical transition (offset). Defaults to 1. Range:[0:20:.1]
    """
    # s = 3 # l is the horizontal transition
    # o = 1 # o is the vertical transition (offset)

    s = s/pi

    selection = {
    "floor"       : lambda phi: floor(m*phi/(2*pi)) + s,  

    "abs"         : lambda phi: abs(m*phi/(2*pi)) + s,  
    
    "rombus?"     : lambda phi: a / (cos(pi*m/4 - (phi % (2*pi*m/4)))),

    "sea star"    : lambda phi: o + a/2 * (cos(m*phi) + 1),

    "sunflower"   : lambda phi: o + a*2 * abs(m*phi/(2*pi) - floor(m*phi/(2*pi) + 1/2)) ,

    "risingsun"   : lambda phi: o + a/2 * (sign(sin(m*phi)) + 1) ,

    "trapez wave" : lambda phi: o + a   * ((arccos(cos(m*phi+s)) + arcsin(sin(m*phi+s)))/pi) ,

    "superformula": lambda phi: (abs(cos(phi*m/4 + s)/a)**n_2 + abs(sin(phi*m/4 + s )/b)**n_3 ) ** (-1/n_1)

    }

    # mod = 1
    #mod = lambda phi : abs(cos(phi*m))
    #r = mod(r)
    
    if funcname is None or funcname not in selection.keys():
        # return all available options of the funcname
        return list(selection.keys())

    chosen = selection[funcname]

    def func(phi):

        r = chosen(phi)
        return r
    return func



def formula2(funcname:str = "superformula",
                m: int = 3,
                a: float = 1, 
                n_1: float = 0.7,
                n_2: float = 2,
                s: float = 0,
                o: float = 1
                ):

    """passes parameters

    Args:
        funcname (str, optional): Function name. Defaults to superformula. Choice:[0:0:0].
        m (int, optional): number of lobes. Defaults to 3. Range:[0:10:.1].
        a (float, optional): Amplitude. Defaults to 1. Range:[.5:10:0.5].
        n_1 (float, optional): Shape parameter 1. Defaults to -1. Dec:[0.01:1000:0.01].
        n_2 (float, optional): Shape parameter 2. Defaults to 2. Dec:[0.01:1000:.01].
        s (float, optional): horizontal transition. Defaults to 0. Range:[-360:360:1]
        o (float, optional): vertical transition (offset). Defaults to 1. Range:[0:20:.1]
    """

    b = a 
    n_3 = n_2

    return formula1(funcname, m, a, b, n_1, n_2, n_3, s, o)