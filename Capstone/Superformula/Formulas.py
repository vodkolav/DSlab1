


from numpy import pi, sin, cos, arcsin, arccos, sign, abs, floor




def parametrize(o: float = 10, # offset
                a: float = 5, # amplitude
                b = 1, # another amplitude
                m: int = 3, # number of lobes
                n1 = -1,
                n2 = 2,
                n3 = 1/2,
                ):
    """passes parameters

    Args:
        o (int, optional): Offset. Defaults to 10. Range:[1:10:0.5].
        n2 (int, optional): _description_. Defaults to 2.
        n3 (_type_, optional): _description_. Defaults to 1/2.
    """
    def sf(phi):

        #l =2.21

        # sea star
        r = o + a * cos(m*phi)

        # sunflower
        k = m*phi/(2*pi)

        # r = o + a * abs(k - floor(k + 1/2))*2
        #r = abs(k - floor(k + 1/(2)))*2
        #r = arccos(cos(m*phi+l))/pi


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

        # superformula
        #r = (abs(cos(phi*m/4)/a)**n2 + abs(sin(phi*m/4)/b)**n3 ) ** (-1/n1)

        return r
    return sf
