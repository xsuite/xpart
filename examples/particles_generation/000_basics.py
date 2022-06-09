# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xobjects as xo

ctx = xo.ContextCpu()

particles = xp.Particles(_context=ctx,
        mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
        x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
        zeta=[1e-2, 2e-2], delta=[0, 1e-4])
