# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xobjects as xo

# Build a reference particle
p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3)

# Choose a context
ctx = xo.ContextCpu()

# Built a set of three particles with different y coordinates
particles = xp.build_particles(_context=ctx, particle_ref=p0, y=[1,2,3])

# Inspect
print(particles.p0c[1]) # gives 7e12
print(particles.x[1]) # gives 0.0
print(particles.y[1]) # gives 2.0

#!end-doc-part

assert particles.p0c[1] == 7e12
assert particles.x[1] == 0.0
assert particles.y[1] == 2.0

