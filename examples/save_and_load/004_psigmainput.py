# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xtrack as xt

cav = xt.Cavity(frequency=400e6, voltage=6e6)

part = xp.Particles(p0c=6500e9, x=[1,2,3], delta=1e-4)
cav.track(part)

part2 = xp.Particles(**part.to_dict())

assert np.all(part2.ptau == part.ptau)
assert np.all(part2.delta == part.delta)
assert np.all(part2.rpp == part.rpp)
assert np.all(part2.rvv == part.rvv)

