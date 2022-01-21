import numpy as np
import xpart as xp
import xobjects as xo

def test_basics():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context,
                mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
                x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
                zeta=[1e-2, 2e-2], delta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert dct['psigma'][0] == 0
        assert np.isclose(dct['psigma'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

def test_unallocated_particles():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context, _capacity=10,
        mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
        x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
        zeta=[1e-2, 2e-2], delta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert dct['psigma'][0] == 0
        assert np.isclose(dct['psigma'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

        particles2 = xp.Particles.from_dict(dct, _context=context)


def test_linked_arrays():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        ctx2np = context.nparray_from_context_array
        particles = xp.Particles(_context=context, p0c=26e9, delta=[1,2,3])

        assert ctx2np(particles.delta[2]) == 3
        assert np.isclose(ctx2np(particles.rvv[2]), 1.00061, rtol=0, atol=1e-5)
        assert np.isclose(ctx2np(particles.rpp[2]), 0.25, rtol=0, atol=1e-10)
        assert np.isclose(ctx2np(particles.psigma[2]), 3.001464, rtol=0, atol=1e-6)

        particles.delta[1] = particles.delta[2]

        assert particles.delta[2] == particles.delta[1]
        assert particles.psigma[2] == particles.psigma[1]
        assert particles.rpp[2] == particles.rpp[1]
        assert particles.rvv[2] == particles.rvv[1]

