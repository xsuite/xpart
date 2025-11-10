# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xobjects as xo
import xtrack as xt

# Get a line
line = xt.load('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')

# Attach a reference particle to the line
line.set_particle_ref('proton', p0c=7e12, x=1, y=3)

# Built a set of three particles with different y coordinates
# (context and particle_ref are taken from the line)
particles = line.build_particles(y=[1,2,3])

#!end-doc-part

assert particles.p0c[1] == 7e12
assert particles.x[1] == 0.0
assert particles.y[1] == 2.0

