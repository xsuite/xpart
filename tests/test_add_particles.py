# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts(excluding='ContextPyopencl')
def test_add_particles(test_context):
    capacity = 10

    p_base = xp.Particles(x=[1, 2, 3], delta=[1e-3, 2e-2, 3e-3], p0c=7e12,
                          _capacity=capacity,
                          mass0=xp.ELECTRON_MASS_EV,
                         _context=test_context)
    
    p1 = p_base.copy()
    p2 = p_base.copy()
    
    p1._init_random_number_generator()

    custom_seeds = np.linspace(10, 20, capacity).astype(int)
    p2._init_random_number_generator(seeds=custom_seeds)

    p_add = xp.Particles(
        x=[4, 5], p0c=7e12, mass0=xp.ELECTRON_MASS_EV, _context=test_context)
    # To test this doesn't override the base seeds
    p_add._init_random_number_generator(seeds=[60, 70])

    p1_check = p1.copy()
    p2_check = p2.copy()
    p1.add_particles(p_add, keep_lost=False)
    p2.add_particles(p_add, keep_lost=False)


    tocpu = test_context.nparray_from_context_array
    # Check parameters that need to be updated
    for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
        assert np.all(tocpu(getattr(p1, nn)[0:3]) ==
                      p_base.to_dict()[nn][0:3])
        assert np.all(tocpu(getattr(p1, nn)[3:5]) ==
                      p_add.to_dict()[nn][0:2])
        
    # Check that the seeds (allocated up to capacity) are not overwritten
    for nn in '_rng_s1 _rng_s2 _rng_s3'.split():
        assert np.all(tocpu(getattr(p1, nn)) ==
                      p1_check.to_dict()[nn])
        
        assert np.all(tocpu(getattr(p2_check, nn)) ==
                      p2_check.to_dict()[nn])