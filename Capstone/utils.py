


isSubscriptable = lambda obj: hasattr(obj, '__getitem__')

def shape(item):
    if hasattr(item, '__getitem__'):
        if hasattr(item, 'shape'):
            return item.shape
        else:
            if hasattr(item, '__len__'):
                return len(item)
            else:
                raise ValueError("Item has __getitem__ but no shape or length.")
    else:
        return item

def describe(subst):
    fmt = "{n} {s}: {t}"
    summ = [fmt.format(n=k, t=type(v), s= shape(v) ) for k,v in subst.items() ]
    print(*summ, sep='\n')


def clip(vars, indcs = slice(None)):

    rooster = vars.split(", ")
    #print(rooster)
    thevars = {k: globals()[k] for k in rooster}

    subst = {k: v[indcs] if isSubscriptable(v) else v  for k, v in thevars.items()}
    describe(subst)
    return subst

def pick(dt):
    fields = "I, step, X, Y, Nx, Ny, circ".split(', ')
    toplot = {f: dt[f] for f in fields}
    return toplot


def trynumeric(val):
    try:
        r = float(val)
    except:
        return val
    return float(val)


def  parse_sf_params(fn):
    """Parse sf parameters from saved graph filename, which usually looks like:
    fn = "superformula_funcname=risingsun_m=5_a=1_b=1_n_1=5.01_n_2=100_n_3=100_20260427_123859.png"
 
    Args:
        fn (string): the filename to parse

    Returns:
        dict: dictionary of parameters extracted from the filename
    """
    import re
    fn = fn.replace("_", " ")
    fn = re.sub("([a-z]) (\d)=", "\\1_\\2=", fn )
    j = fn.split(" ")
    k = [i.split("=") for i in j if "=" in i]
    l = {i[0]: trynumeric(i[1]) for i in k}
    return l 
