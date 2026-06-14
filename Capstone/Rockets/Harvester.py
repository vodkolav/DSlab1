from copy import deepcopy
import warnings
import numpy as np


class Harvester:
    """Collects variable values across iterations of a loop, and concatenates them at the end."""

    def __init__(self, varnames="", elems=None, on_size_mismatch = 'error'):
        # TODO: add option to specify how to treat error when variable is not found in fnlocals, e.g ignore, warn, or raise error
        self.varnames = varnames
        self.elems = elems
        self.on_size_mismatch = on_size_mismatch
        self.storage = {vn:[] for vn in varnames}


    def collect(self, fnlocals):
        #TODO: validate that all variables that we want to collect are of the same size. 
        # Or size 1 - for scalar metrics such as circumference at step, etc.
        size = 1 
        scalars = {}
        for vn in self.varnames:
            varbl = fnlocals[vn]

            # if var is scalar - put it aside to later add as repeated
            if np.isscalar(varbl):
                scalars[vn] = varbl
            else:
                # Extract specific elements if needed
                if self.elems is not None:
                    elems = fnlocals[self.elems]
                    content = varbl[elems]
                else:
                    content = varbl

                size = self.validate_size(size, content)
                
                self.storage[vn].append(deepcopy(content))
        
        for vn, content in scalars.items():
            self.storage[vn].append(np.repeat(content, size))


    def validate_size(self, size, content):
        if self.on_size_mismatch == 'ignore':
            return size
        
        thissize = len(content)
        if size!=1: 
            if thissize != size: 
                if self.on_size_mismatch == 'warn':
                    warnings.warn("Mismatch in collected variables sizes") 
                elif self.on_size_mismatch == 'error':
                    raise ValueError("All variables within particular collect() call must be either of the same size or scalars ")
        else: 
            size = thissize

        return size


    def add_once(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.storage.keys():
                self.storage[k].append(deepcopy(v))
            else:
                self.storage[k] = [deepcopy(v)]

    def results(self, **kwargs):
        store = {vn: np.concatenate(vals,axis=0) for vn, vals in self.storage.items()}

        if kwargs:
            # kwargs = {"S":(5,6)}
            for k,v in kwargs.items():
                #TODO: Support multiple conditions with and/or
                mask = np.isin(store[k], v)
                outpt = {vn: vl[mask] for vn,vl in store.items()} 
            return outpt
        
        return store