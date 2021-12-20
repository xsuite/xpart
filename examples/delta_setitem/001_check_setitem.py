import numpy as np

import xpart as xp

particles = xp.Particles(p0c=26e9, delta=[1,2,3])

assert particles.delta[2] == 3
assert np.isclose(particles.rvv[2], 1.00061, rtol=0, atol=1e-5)
assert np.isclose(particles.rpp[2], 0.25, rtol=0, atol=1e-10)
assert np.isclose(particles.psigma[2], 3.001464, rtol=0, atol=1e-6)

particles.delta[1] = particles.delta[2]

assert particles.delta[2] == particles.delta[1]
assert particles.psigma[2] == particles.psigma[1]
assert particles.rpp[2] == particles.rpp[1]
assert particles.rvv[2] == particles.rvv[1]