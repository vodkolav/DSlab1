


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
