import numpy as np
import xpart as xp
import xobjects as xo

def test_basics():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context,
                mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
                x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
                z=[1e-2, 2e-2], delta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert dct['psigma'][0] == 0
        assert np.isclose(dct['psigma'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)
