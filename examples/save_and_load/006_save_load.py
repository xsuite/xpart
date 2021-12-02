import numpy as np

import xobjects as xo
import xpart as xp

# Create a Particles on your selected context (default is CPU)
context = xo.ContextCupy()
part = xp.Particles(_context=context, x=[1,2,3])

########
# JSON #
########

# Save particles to json
import json
with open('part.json', 'w') as fid:
    json.dump(part.to_dict(), fid, cls=xo.JEncoder)

# Load particles from json file to selected context
with open('part.json', 'r') as fid:
    part_from_json= xp.Particles.from_dict(json.load(fid), _context=context)

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

##############
# PANDAS/HDF #
##############

# Save particles to hdf file via pandas
import pandas as pd
df = part.to_pandas()
df.to_hdf('part.hdf', key='df', mode='w')

# Read particles from hdf file via pandas
part_from_pdhdf = xp.Particles.from_pandas(pd.read_hdf('part.hdf'))

#!end-doc-part
dct = part.to_dict()
assert isinstance(dct['delta'], np.ndarray)
