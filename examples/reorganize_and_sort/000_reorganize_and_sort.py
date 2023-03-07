# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xpart as xp
import xtrack as xt

p = xp.Particles(x=[0, 1, 2, 3, 4, 5, 6], _capacity=10)
p.state[[0,3,4]] = 0

line=xt.Line(elements=[xt.Cavity()])
line.build_tracker()
line.track(p)

assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.x == np.array([6, 1, 2, 5, 4, 3, 0,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == 4
assert p._num_lost_particles == 3

p.sort()

assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == 4
assert p._num_lost_particles == 3

p.sort(interleave_lost_particles=True)

assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([0, 1, 1, 0, 0, 1, 1,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == -2
assert p._num_lost_particles == -2

p = xp.Particles(x=[6, 5, 4, 3, 2, 1, 0], _capacity=10)
p.state[[0,3,4]] = 0

line.track(p)

assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.x == np.array([0, 5, 4, 1, 2, 3, 6,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == 4
assert p._num_lost_particles == 3

p.sort(by='x')

assert np.all(p.particle_id == np.array([6, 5, 2, 1, 4, 3, 0,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.x == np.array([0, 1, 4, 5, 2, 3, 6,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == 4
assert p._num_lost_particles == 3

p.sort(by='x', interleave_lost_particles=True)

assert np.all(p.particle_id == np.array([6, 5, 4, 3, 2, 1, 0,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.x == np.array([0, 1, 2, 3, 4, 5, 6,
                                  -999999999, -999999999, -999999999]))
assert np.all(p.state == np.array([1, 1, 0, 0, 1, 1, 0,
                                  -999999999, -999999999, -999999999]))
assert p._num_active_particles == -2
assert p._num_lost_particles == -2
