# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_basics(test_context):
    p1 = xp.Particles(x=[1, 2, 3], px=[10, 20, 30],
                      mass0=xp.ELECTRON_MASS_EV,
                      _context=test_context)

    mask = p1.x > 1

    p2 = p1.filter(mask)

    assert p2._buffer.context == test_context
    assert p2._capacity == 2
    dct = p2.to_dict()
    assert dct['mass0'] == xp.ELECTRON_MASS_EV
    assert np.all(dct['px'] == np.array([20., 30.]))
