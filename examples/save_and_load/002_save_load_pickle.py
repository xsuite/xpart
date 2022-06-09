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

##########
# PICKLE #
##########

# Save particles to pickle file
import pickle
with open('part.pkl', 'wb') as fid:
    pickle.dump(part.to_dict(), fid)

# Load particles from json to selected context
with open('part.pkl', 'rb') as fid:
    part_from_pkl= xp.Particles.from_dict(pickle.load(fid), _context=context)
