import inspect
from typing import Any
from pydoc import locate
from docstring_parser import parse, DocstringParam
import re

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

    print(Desc, Rng)
    spl = re.findall("[\s\w]+:\[([\d\.]+):?([\d\.]+)?:?([\d\.]+)?\]",Rng)
    print(spl)
    Min, Max, Step = spl[0]

    p = {
    "Desc": Desc, "Name":prm.name, "Type": Type, 
    "Min":float(Min), "Max":float(Max), "Step":float(Step), "Deflt": prm.default
    }
    return p


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

