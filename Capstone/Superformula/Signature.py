import inspect
from typing import Any
from pydoc import locate
from docstring_parser import parse, DocstringParam
import re
import numpy as np

def treat_param(prm: inspect.Parameter, doc: DocstringParam ):
    """Extract all available information on a parameter via inspect and docstring 

    Args:
        prm (inspect.Parameter): the inspect object for param
        doc (DocstringParam): the docstring object for param

    Returns:
        dict: extracted information
    """

    if doc is None:
        Desc, Deft, Rng = "N/A", "1", " Range:[0:1:0.1]" #description, default value, default range
    else:
        desc = re.split("[.;](?:\s+|$)",doc.description)
        Desc, Deft, Rng  = desc[:3]
        if Rng == '':
            Rng = " Range:[0:1:0.1]" 

    if prm.annotation is inspect._empty:
        if doc and doc.type_name:
            Type = locate(doc.type_name)
        else:
            Type = Any
    else:
        Type = prm.annotation

    #print(Desc, Rng)
    spl = re.findall("([\s\w]+):\[([-\d\.]+):?([\d\.]+)?:?([\d\.]+)?\]",Rng)
    #print(spl)
    Scl, Min, Max, Step = spl[0]

    # Deft = 10 ** prm.default if Scl == "Dec" else prm.default
    Deft  = prm.default

    p = {
    "Desc": Desc, "Name":prm.name, "Type": Type, "Scl": Scl, "Opts": None,
    "Min":float(Min), "Max":float(Max), "Step":float(Step), "Deflt": Deft
    }
    return p


def dec_scale(min,max):
    # attempt to fix marks at integer * decimal points, 
    # so that they look nice: 
    # 0.1, 0.2, 0.3, ... 1, 2, 3, ... 10, 20, 30, ...
    # needs more work
    
    a = np.arange(float(min),float(max))
    s = a.shape[0]
    b = 10 ** a
    b = np.expand_dims(b,1)
    
    cc = np.arange(0,10)
    c  = np.expand_dims(np.arange(1,10),0)

    d  = np.dot(b,c+1)
    sm = np.add.outer(a*10,cc)

    e  =  d.reshape(1,9*s).squeeze(0)
    sm = sm.reshape(1,10*s).squeeze(0)[:9*s]

    return sm, e

def dec_scale_2(min,max,step):
    a = np.arange(min, max+step, step)
    marks_desired = 11
    stp = int(np.floor(a.shape[0] / marks_desired))
    ass = a[::stp]
    bss = 10 ** ass
    return ass, bss


def exp_scale(mn, mx, num):
    """Produces an exponentially scaled axis for the slider control.
        When you need to control a variable with possible values in a wide range of orders of magnitude.
        e.g.  0.0001 < x < 10000

    Args:
        mn (float): minimal value of the scale. also defines minimal order of magnitude 
        mx (float): maximal value of the scale. also defines maximal order of magnitude
        num (int): the amount of marks to produce

    Returns:
        float, float: ax - the linear values for the slider; val - the exponential values for the parameter
    """
    s = np.sign(mn)
    lmin = np.log10(np.abs(mn)) if mn != 0 else 0
    lmax = np.log10(np.abs(mx)) if mx != 0 else 0
    ax, stp = np.linspace(s * lmin, lmax, num=num, retstep=True)
    sg = np.sign(ax)
    # val = sg * (10 ** np.abs(ax))
    val = 10 ** ax
    return ax, val


def to_si(val, precision=2):
    """Format floats of wide range of orders of magnitudes to be human readable and compact

    Args:
        val (float): the value to format
        precision (int, optional): unused. Defaults to 2.

    Returns:
        str: the formatted string representation
    """
    ans = 0
    if val == 0: 
        return "0"
        

    # Standard SI prefixes (10^3, 10^6, 10^9, etc.)
    prefixes = ["n", "μ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"]
    #    array([-3., -2., -1., 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    powers   = np.arange(-3.0, 9.0) 
    #powers[2] = 0

    s = int(np.floor(np.log10(abs(val))))
    # Calculate which index in 'prefixes' to use
    j = int(np.floor(np.log10(abs(val)) / 3))
    # print(i)
    # i = np.max([0, np.min([j, len(prefixes) - 1])]) # stay within bounds
    i = np.where(powers==j)[0][0] if j in powers else 1

    #idx = np.where(powers==j)
    # idx = i
    # print(i)
    idx = 3 if s in [-1,] else i
    scaled = val / (1000 ** powers[idx])
    # print(s)
    # Format with chosen precision and strip trailing zeros
    fmt = f"{scaled:2.1f}"
    # print(s, i, fmt)
    ans = fmt.rstrip('0').rstrip('.') + prefixes[idx]
    #return ans
    full = f"{val:.10f}"

    return ans

def analyze_function(thefunc):
    """Extract all available information on parameters of thefunc via inspect and docstring 

    Args:
        thefunc (callable): the function to be analyzed

    Returns:
        dict: the extractred information on the params
    """
    signs = inspect.signature(thefunc)

    # Parse the docstring
    docs = parse(thefunc.__doc__)
    docs = {pr.arg_name: pr for pr in docs.params}

    pars = {}

    for param in signs.parameters.values():
        name = param.name
        doc = docs.get(name,None)
        p = treat_param(param, doc)

        if p["Scl"] == "Choice":
            p["Opts"] = thefunc(funcname=None)

        pars[param.name] = p

    return pars

