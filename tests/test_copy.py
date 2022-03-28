import numpy as np

import xpart as xp
import xobjects as xo

def test_copy():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        p1 = xp.Particles(x=[1,2,3], delta=1e-3, _context=context)

        # Make a copy of p1 in the same context
        p2 = p1.copy()

        # Copy across contexts
        p3 = p1.copy(_context=xo.ContextCpu())

        # And back
        p4 = p3.copy(_context=context)

        dct1 = p1.to_dict()
        dct2 = p2.to_dict()
        dct3 = p2.to_dict()
        dct4 = p3.to_dict()
        for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
            for dct in [dct2, dct3, dct4]:
                assert np.all(dct[nn] == dct1[nn])

