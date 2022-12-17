import numpy as np
import xpart as xp
import xobjects as xo

def test_reorganize_particles():

    for context in xo.context.get_test_contexts():
        print(f"Testing with context {context}")

        c2n = context.nparray_from_context_array
        n2c = context.nparray_to_context_array

        particles = xp.Particles(_context=context, p0c=450e9, x=[1,2,3,4,5],
                                delta=[0.1,0.2,0.3,0.4,0.5])
        particles0 = particles.copy()

        assert np.all(c2n(particles.state) == 1)

        n_act, n_lost = particles.reorganize()
        assert n_act == 5

        particles.state = n2c(np.array([1,1,0,0,1]))
        n_act, n_lost = particles.reorganize()
        assert n_act == 3
        assert n_lost == 2

        assert np.all(c2n(particles.x[0:3]) == [1,2,5])
        assert np.all(c2n(particles.delta[0:3]) == [0.1,0.2,0.5])
        assert np.all(c2n(particles.ptau[0:3]) == c2n(particles0.ptau)[np.array([0,1,4])])

def test_hide_lost_particles():

    for context in xo.context.get_test_contexts():
        print(f"Testing with context {context}")

        c2n = context.nparray_from_context_array
        n2c = context.nparray_to_context_array

        particles = xp.Particles(p0c=450e9, x=[1,2,3,4,5],
                                delta=[0.1,0.2,0.3,0.4,0.5])
        particles0 = particles.copy()

        assert np.all(c2n(particles.state) == 1)

        particles.hide_lost_particles()
        assert(len(particles.x) == 5)
        particles.unhide_lost_particles()
        assert(len(particles.x) == 5)

        particles.state = n2c(np.array([1,1,0,0,1]))
        particles.hide_lost_particles()
        assert(len(particles.x) == 3)
        assert np.all(c2n(particles.x) == [1,2,5])
        assert np.all(c2n(particles.delta) == [0.1,0.2,0.5])
        assert np.all(c2n(particles.ptau) == c2n(particles0.ptau)[np.array([0,1,4])])

        particles.unhide_lost_particles()
        assert(len(particles.x) == 5)
        assert np.all(c2n(particles.x[0:3]) == [1,2,5])
        assert np.all(c2n(particles.delta[0:3]) == [0.1,0.2,0.5])
        assert np.all(c2n(particles.ptau[0:3]) == c2n(particles0.ptau)[np.array([0,1,4])])