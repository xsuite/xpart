# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xobjects as xo

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_build_particles_shift(test_context):
    for ctx_ref in [test_context, None]:
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[1e-4], _context=ctx_ref)


        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(mode='shift', particle_ref=p0, y=[1,2,3],
                                       _context=test_context)

        dct = particles.to_dict() # transfers it to cpu
        assert np.isclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)
        assert np.all(dct['p0c'] == 7e12)
        assert dct['x'][1] == 1.0
        assert dct['y'][1] == 5.0
