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

##############
# PANDAS/HDF #
##############

# Save particles to hdf file via pandas
import pandas as pd
df = part.to_pandas()
df.to_hdf('part.hdf', key='df', mode='w')

# Read particles from hdf file via pandas
part_from_pdhdf = xp.Particles.from_pandas(pd.read_hdf('part.hdf'))
