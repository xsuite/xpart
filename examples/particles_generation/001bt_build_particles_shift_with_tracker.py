# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xobjects as xo
import xtrack as xt

# Load machine model
line = xt.load('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')

# Attach a reference particle to the line
line.set_particle_ref('proton', p0c=7e12, x=1, y=3)

# Built a set of three particles with different y coordinates
# (context and particle_ref are taken from the line)
particles = line.build_particles(mode='shift', y=[1,2,3])

# Inspect
print(particles.p0c[1]) # gives 7e12
print(particles.x[1]) # gives 1.0
print(particles.y[1]) # gives 5.0

#!end-doc-part

assert particles.p0c[1] == 7e12
assert particles.x[1] == 1.0
assert particles.y[1] == 5.0

