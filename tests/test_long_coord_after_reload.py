import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

def test_to_from_dict_longitudinal_consistency():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        cav = xt.Cavity(_context=context, frequency=400e6, voltage=6e6)

        part = xp.Particles(_context=context, p0c=6500e9, x=[1,2,3], delta=1e-4)
        cav.track(part)

        part2 = xp.Particles(_context=context, **part.to_dict())

        tocpu = context.nparray_from_context_array
        for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
            assert np.all(tocpu(getattr(part, nn)) == tocpu(getattr(part2, nn)))

def test_to_from_pandas_longitudinal_consistency():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        cav = xt.Cavity(_context=context, frequency=400e6, voltage=6e6)

        part = xp.Particles(_context=context, p0c=6500e9, x=[1,2,3], delta=1e-4)
        cav.track(part)

        df = part.to_pandas()
        part2 = xp.Particles.from_pandas(df, _context=context)

        tocpu = context.nparray_from_context_array
        for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
            assert np.all(tocpu(getattr(part, nn)) == tocpu(getattr(part2, nn)))
