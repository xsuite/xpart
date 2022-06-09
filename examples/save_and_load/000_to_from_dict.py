# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp

# Create a Particles on your selected context (default is CPU)
context = xo.ContextCupy()
part = xp.Particles(_context=context, x=[1,2,3])

# Save particles to dict 
dct = part.to_dict()

# Load particles from dict 
part_from_dict = xp.Particles.from_dict(dct, _context=context)

#!end-doc-part
dct = part.to_dict()
assert isinstance(dct['delta'], np.ndarray)
