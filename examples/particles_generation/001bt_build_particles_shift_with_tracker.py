# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import json

import xpart as xp
import xobjects as xo
import xtrack as xt

ctx = xo.ContextCpu() # choose a context

# Load machine model and built a line
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    line = xt.Line.from_dict(json.load(fid)['line'])
line.build_tracker(_context=ctx)

# Attach a reference particle to the line
line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, q0=1, x=1 , y=3)

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

