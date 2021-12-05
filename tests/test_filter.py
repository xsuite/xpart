import numpy as np

import xobjects as xo
import xpart as xp

def test_basics():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        p1 = xp.Particles(x=[1,2,3], px=[10, 20, 30],
                          mass0=xp.ELECTRON_MASS_EV,
                          _context=context)

        mask = p1.x > 1

        if isinstance(context, xo.ContextPyopencl):
            mask = context.nparray_from_context_array(mask)>0

        p2 = p1.filter(mask)

        assert p2._buffer.context == context
        assert p2._capacity == 2
        dct = p2.to_dict()
        assert dct['mass0'] == xp.ELECTRON_MASS_EV
        assert np.all(dct['px'] == np.array([20., 30.]))
