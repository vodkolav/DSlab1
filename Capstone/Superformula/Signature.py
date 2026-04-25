import inspect
from typing import Any
from pydoc import locate
from docstring_parser import parse, DocstringParam
import re
import numpy as np

def treat_param(prm: inspect.Parameter, doc: DocstringParam ):
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

    p = {
    "Desc": Desc, "Name":prm.name, "Type": Type, "Scl": Scl,
    "Min":float(Min), "Max":float(Max), "Step":float(Step), "Deflt": prm.default
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


def analyze_function(thefunc):

    signs = inspect.signature(thefunc)

    # Parse the docstring
    docs = parse(thefunc.__doc__)
    docs = {pr.arg_name: pr for pr in docs.params}

    pars = {}

    for param in signs.parameters.values():
        name = param.name
        doc = docs.get(name,None)
        pars[param.name] = treat_param(param, doc)

    return pars

#funcparams = analyze_function(parametrize)

