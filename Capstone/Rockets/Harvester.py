from copy import deepcopy

import numpy as np


class Harvester:
    """Collects variable values across iterations of a loop, and concatenates them at the end."""

    def __init__(self, varnames="", elems=None):
        self.varnames = varnames
        self.elems = elems
        self.storage = {vn:[] for vn in varnames}


    def collect(self, fnlocals):
        for vn in self.varnames:
            varbl = fnlocals[vn]
            if self.elems is not None:
                elems = fnlocals[self.elems]
                content = varbl[elems]
            else:
                content = varbl
            self.storage[vn].append(deepcopy(content))

    def results(self):
        return {vn: np.concatenate(vals,axis=0) for vn, vals in self.storage.items()}