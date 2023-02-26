# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xobjects as xo

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_copy(test_context):
    p1 = xp.Particles(x=[1, 2, 3], delta=1e-3, _context=test_context)

    # Make a copy of p1 in the same context
    p2 = p1.copy()

    # Copy across contexts
    p3 = p1.copy(_context=xo.ContextCpu())

    # And back
    p4 = p3.copy(_context=test_context)

    dct1 = p1.to_dict()
    dct2 = p2.to_dict()
    dct3 = p3.to_dict()
    dct4 = p4.to_dict()
    for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
        for dct in [dct2, dct3, dct4]:
            assert np.all(dct[nn] == dct1[nn])
