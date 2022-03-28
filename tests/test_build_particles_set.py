import numpy as np

import xpart as xp
import xobjects as xo

def test_build_set():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        for ctx_ref in [ctx, None]:
            # Build a reference particle
            p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                              delta=[10], _context=ctx_ref)

            # Built a set of three particles with different x coordinates
            particles = xp.build_particles(_context=ctx,
                                           particle_ref=p0, y=[1,2,3],
                                           delta=[0, 1e-4, -1e-4])

            dct = particles.to_dict() # transfers it to cpu
            assert dct['ptau'][0] == 0
            assert np.isclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
            assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)
            assert np.all(dct['p0c'] == 7e12)
            assert dct['x'][1] == 0.0
            assert dct['y'][1] == 2.0

