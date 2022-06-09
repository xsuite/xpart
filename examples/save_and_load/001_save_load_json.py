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

# Save particles to json
import json
with open('part.json', 'w') as fid:
    json.dump(part.to_dict(), fid, cls=xo.JEncoder)

# Load particles from json file to selected context
with open('part.json', 'r') as fid:
    part_from_json= xp.Particles.from_dict(json.load(fid), _context=context)

