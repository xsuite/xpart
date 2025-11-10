# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo
import xpart as xp
import xtrack as xt

line = xt.load('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')

# Attach a reference particle to the line
line.set_particle_ref('proton', p0c=7e12, x=1, y=3)

# Built a set of three particles with different x coordinates
particles = line.build_particles(
                               zeta=0, delta=1e-3,
                               x_norm=[1,0,-1], # in sigmas
                               px_norm=[0,1,0], # in sigmas
                               nemitt_x=3e-6, nemitt_y=3e-6)

#!end-doc-part
