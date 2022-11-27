# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import json

import xpart as xp
import xobjects as xo
import xtrack as xt

# Load machine model and built a tracker
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, q0=1,
                                x =1 , y=3)

ctx = xo.ContextCpu() # choose a context
tracker = line.build_tracker(_context=ctx)

# Built a set of three particles with different y coordinates
# (context and particle_ref are taken from the tracker)
particles = tracker.build_particles(y=[1,2,3])

# Inspect
print(particles.p0c[1]) # gives 7e12
print(particles.x[1]) # gives 0.0
print(particles.y[1]) # gives 2.0

#!end-doc-part

assert particles.p0c[1] == 7e12
assert particles.x[1] == 0.0
assert particles.y[1] == 2.0

