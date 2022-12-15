import numpy as np
import xpart as xp

def test_reorganize_particles():

    particles = xp.Particles(p0c=450e9, x=[1,2,3,4,5],
                             delta=[0.1,0.2,0.3,0.4,0.5])
    particles0 = particles.copy()

    assert np.all(particles.state == 1)

    n_act, n_lost = particles.reorganize()
    assert n_act == 5

    particles.state = np.array([1,1,0,0,1])
    n_act, n_lost = particles.reorganize()
    assert n_act == 3
    assert n_lost == 2

    assert np.all(particles.x[0:3] == [1,2,5])
    assert np.all(particles.delta[0:3] == [0.1,0.2,0.5])
    assert np.all(particles.ptau[0:3] == particles0.ptau[np.array([0,1,4])])