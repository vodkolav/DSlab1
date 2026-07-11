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


    def get(self, fnlocals, vn):
        """if var name contains a dot, it is assumed to be a var.attribute.
        Extract this attribute

        Args:
            fnlocals (_type_): _description_
            vn (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if ("." in vn):
            pth = vn.split(".")
            if len(pth) > 2:
                raise ValueError("attributes of depth more than 2 not supported yet")    
            var, atr = pth
            varbl = fnlocals[var].__getattribute__(atr)
            return varbl
        else:
            return fnlocals[vn]


    def collect(self, fnlocals):

        #TODO: validate that all variables that we want to collect are of the same size. 
        # Or size 1 - for scalar metrics such as circumference at step, etc.
        size = 1 
        scalars = {}
        for vn in self.varnames:

            varbl = self.get(fnlocals,vn)

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

    def results(self, extract_attrs = True, **kwargs):
        """Returns the harvested data

        Args:
            extract_attrs (bool, optional): rename var.attribute data to just attribute. Defaults to True.

        Returns:
            _type_: _description_
        """

        e = lambda k: k.split(".")[-1] if extract_attrs else k

        store = {e(vn): np.concatenate(vals,axis=0) for vn, vals in self.storage.items()}

        if kwargs:
            # kwargs = {"S":(5,6)}
            for k,v in kwargs.items():
                #TODO: Support multiple conditions with and/or
                mask = np.isin(store[k], v)
                outpt = {vn: vl[mask] for vn,vl in store.items()} 
            return outpt
        
        return store