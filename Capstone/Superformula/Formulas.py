


from numpy import pi, sin, cos, arcsin, arccos, sign, abs, floor



def formula1(m: int = 3,
                a: float = 1, 
                b: float = 1, 
                n1: float = 0.7,
                n2: float = 2,
                n3: float = 2,
                ):

    """passes parameters

    Args:
        m (int, optional): number of lobes. Defaults to 3. Range:[0:10:.1].
        a (float, optional): Amplitude. Defaults to 1. Range:[.5:10:0.5].
        b (float, optional): Another Amplitude. Defaults to 1. Range:[.5:10:0.5].
        n1 (float, optional): Shape parameter 1. Defaults to -1. Dec:[-1:3:0.01].
        n2 (float, optional): Shape parameter 2. Defaults to 2. Dec:[-1:3:.01].
        n3 (float, optional): Shape parameter 3. Defaults to 2. Dec:[-1:3:.01].
    """

    def sf(phi):

        #l =2.21

        # sea star
        #r = o + a * cos(m*phi)

        # sunflower
        #k = m*phi/(2*pi)

        # r = o + a * abs(k - floor(k + 1/2))*2
        #r = abs(k - floor(k + 1/(2)))*2


        #rombus? 
        #r = 10 / (cos(pi/4 - (phi % (pi/2))))

        # risingsun
        p =  m # period
        #r = o + a * sign(sin(p*phi))+1

        # trapez wave
        #p = 1/m # p is the period

        l = 3 # l is the horizontal transition
        c = 0 # c is the vertical transition
        #r = a*(( arccos(cos(m*phi+l)) + arcsin(sin(m*phi+l)))/pi +1) + o

        # mod = 1
        mod = lambda phi : abs(cos(phi*m))
        # superformula
        r = (abs(cos(phi*m/4)/a)**n2 + abs(sin(phi*m/4)/b)**n3 ) ** (-1/n1)
        r = mod(r)
        return r
    return sf
