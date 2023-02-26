# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import numpy as np
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_reorganize_particles(test_context):
    c2n = test_context.nparray_from_context_array
    n2c = test_context.nparray_to_context_array

    particles = xp.Particles(_context=test_context, p0c=450e9, x=[1, 2, 3, 4, 5],
                             delta=[0.1, 0.2, 0.3, 0.4, 0.5])
    particles0 = particles.copy()

    assert np.all(c2n(particles.state) == 1)

    n_act, n_lost = particles.reorganize()
    assert n_act == 5

    particles.state = n2c(np.array([1, 1, 0, 0, 1]))
    n_act, n_lost = particles.reorganize()
    assert n_act == 3
    assert n_lost == 2

    assert np.all(c2n(particles.x[0:3]) == [1, 2, 5])
    assert np.all(c2n(particles.delta[0:3]) == [0.1, 0.2, 0.5])
    assert np.all(c2n(particles.ptau[0:3]) == c2n(particles0.ptau)[np.array([0, 1, 4])])


@for_all_test_contexts
def test_hide_lost_particles(test_context):
    c2n = test_context.nparray_from_context_array
    n2c = test_context.nparray_to_context_array

    particles = xp.Particles(_context=test_context, p0c=450e9, x=[1, 2, 3, 4, 5],
                             delta=[0.1, 0.2, 0.3, 0.4, 0.5])
    particles0 = particles.copy()

    assert np.all(c2n(particles.state) == 1)

    particles.hide_lost_particles()
    assert(len(particles.x) == 5)
    particles.unhide_lost_particles()
    assert(len(particles.x) == 5)

    particles.state = n2c(np.array([1, 1, 0, 0, 1]))
    particles.hide_lost_particles()
    assert(len(particles.x) == 3)
    assert np.all(c2n(particles.x) == [1, 2, 5])
    assert np.all(c2n(particles.delta) == [0.1, 0.2, 0.5])
    assert np.all(c2n(particles.ptau) == c2n(particles0.ptau)[np.array([0, 1, 4])])

    particles.unhide_lost_particles()
    assert(len(particles.x) == 5)
    assert np.all(c2n(particles.x[0:3]) == [1, 2, 5])
    assert np.all(c2n(particles.delta[0:3]) == [0.1, 0.2, 0.5])
    assert np.all(c2n(particles.ptau[0:3]) == c2n(particles0.ptau)[np.array([0, 1, 4])])
