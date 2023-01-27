# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_to_from_dict_longitudinal_consistency(test_context):
    cav = xt.Cavity(_context=test_context, frequency=400e6, voltage=6e6)

    part = xp.Particles(_context=test_context, p0c=6500e9, x=[1, 2, 3], delta=1e-4)
    cav.track(part)

    part2 = xp.Particles(_context=test_context, **part.to_dict())

    tocpu = test_context.nparray_from_context_array
    for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
        assert np.all(tocpu(getattr(part, nn)) == tocpu(getattr(part2, nn)))


@for_all_test_contexts
def test_to_from_pandas_longitudinal_consistency(test_context):
    cav = xt.Cavity(_context=test_context, frequency=400e6, voltage=6e6)

    part = xp.Particles(_context=test_context, p0c=6500e9, x=[1, 2, 3], delta=1e-4)
    cav.track(part)

    df = part.to_pandas()
    part2 = xp.Particles.from_pandas(df, _context=test_context)

    tocpu = test_context.nparray_from_context_array
    for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
        assert np.all(tocpu(getattr(part, nn)) == tocpu(getattr(part2, nn)))
