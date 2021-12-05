import numpy as np

import xobjects as xo
import xpart as xp

def test_to_from_dict():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        # Create a Particles on your selected context (default is CPU)

        part = xp.Particles(_context=context, x=[1,2,3])

        # Save particles to dict 
        dct = part.to_dict()

        # Load particles from dict 
        part_from_dict = xp.Particles.from_dict(dct, _context=context)

        #!end-doc-part
        dct = part.to_dict()
        for nn in 'x px y py zeta delta psigma rpp rvv gamma0 p0c'.split():
            assert isinstance(dct[nn], np.ndarray)
            assert isinstance(getattr(part_from_dict, nn), context.nplike_array_type)

