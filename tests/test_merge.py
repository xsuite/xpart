# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp

def test_merge():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        p1 = xp.Particles(x=[1,2,3], delta=[1e-3, 2e-2, 3e-3], p0c=7e12,
                mass0=xp.ELECTRON_MASS_EV,
                _context=context)
        p2 = xp.Particles(x=[4, 5], p0c=7e12, mass0=xp.ELECTRON_MASS_EV)
        p3 = xp.Particles(x=6, delta=[-1e-1], p0c=7e12, mass0=xp.ELECTRON_MASS_EV)

        particles = xp.Particles.merge([p1,p2,p3])

        assert particles._buffer.context == context
        assert particles.mass0 == xp.ELECTRON_MASS_EV
        tocpu = context.nparray_from_context_array
        for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
            assert np.all(tocpu(getattr(particles, nn)[0:3]) ==
                          p1.to_dict()[nn])
            assert np.all(tocpu(getattr(particles, nn)[3:5]) ==
                          p2.to_dict()[nn])
            assert np.all(tocpu(getattr(particles, nn)[5:]) ==
                          p3.to_dict()[nn])
