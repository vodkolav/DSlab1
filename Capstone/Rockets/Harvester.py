from copy import deepcopy

import numpy as np


class Harvester:
    """Collects variable values across iterations of a loop, and concatenates them at the end."""

    def __init__(self, varnames="", elems=None):
        # TODO: add option to specify how to treat error when variable is not found in fnlocals, e.g ignore, warn, or raise error
        self.varnames = varnames
        self.elems = elems
        self.storage = {vn:[] for vn in varnames}


    def collect(self, fnlocals):
        #TODO: validate that all variables that we want to collect are of the same size. 
        # Or size 1 - for scalar metrics such as circumference at step, etc.
        for vn in self.varnames:
            varbl = fnlocals[vn]
            if self.elems is not None:
                elems = fnlocals[self.elems]
                content = varbl[elems]
            else:
                content = [varbl] if np.isscalar(varbl) else varbl
            self.storage[vn].append(deepcopy(content))


    def add_once(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.storage.keys():
                self.storage[k].append(deepcopy(v))
            else:
                self.storage[k] = [deepcopy(v)]

    def results(self):
        return {vn: np.concatenate(vals,axis=0) for vn, vals in self.storage.items()}