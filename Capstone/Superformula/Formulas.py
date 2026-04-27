


from numpy import pi, sin, cos, arcsin, arccos, sign, abs, floor



def formula1(funcname:str = "superformula",
                m: int = 3,
                a: float = 1, 
                b: float = 1, 
                n_1: float = 0.7,
                n_2: float = 2,
                n_3: float = 2,
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
    """
    l = 3 # l is the horizontal transition
    o = 1 # o is the vertical transition (offset)

    selection = {
    
    "rombus?"     : lambda phi: a / (cos(pi*m/4 - (phi % (2*pi*m/4)))),

    "sea star"    : lambda phi: o + a * cos(m*phi),

    "sunflower"   : lambda phi: o + a * abs(m*phi/(2*pi) - floor(m*phi/(2*pi) + 1/2))*2 ,

    "risingsun"   : lambda phi: o + a * sign(sin(m*phi))+1 ,

    "trapez wave" : lambda phi: o + a * ((arccos(cos(m*phi+l)) + arcsin(sin(m*phi+l)))/pi +1) ,

    "superformula": lambda phi: (abs(cos(phi*m/4)/a)**n_2 + abs(sin(phi*m/4)/b)**n_3 ) ** (-1/n_1)

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
