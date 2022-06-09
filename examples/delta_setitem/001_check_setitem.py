# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xobjects as xo

#context = xo.ContextPyopencl()
context = xo.ContextCpu()
ctx2np = context.nparray_from_context_array
particles = xp.Particles(_context=context, p0c=26e9, delta=[1,2,3])

assert ctx2np(particles.delta[2]) == 3
assert np.isclose(ctx2np(particles.rvv[2]), 1.00061, rtol=0, atol=1e-5)
assert np.isclose(ctx2np(particles.rpp[2]), 0.25, rtol=0, atol=1e-10)
assert np.isclose(ctx2np(particles.ptau[2]), 3.001464*particles._xobject.beta0[0],
                  rtol=0, atol=1e-6)

particles.delta[1] = particles.delta[2]

assert particles.delta[2] == particles.delta[1]
assert particles.ptau[2] == particles.ptau[1]
assert particles.rpp[2] == particles.rpp[1]
assert particles.rvv[2] == particles.rvv[1]
