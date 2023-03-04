# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #


import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo


import xpart as xp
import xtrack as xt
import time

ctx = xo.ContextCpu()

# Load machine model
filename = xt._pkg_root.parent.joinpath(
    'test_data/sps_ions/line_and_particle.json')
line = xt.Line.from_json(filename)
line.build_tracker(_context=ctx)

rms_bunch_length=0.14
distribution = "gaussian"
n_particles = 10000
zeta, delta, matcher = xp.generate_longitudinal_coordinates(line.line,
                        num_particles=n_particles,
                        sigma_z=rms_bunch_length, distribution=distribution,
                        engine="single-rf-harmonic", return_matcher=True)
                        #engine="pyheadtail", return_matcher=True)

# Built a set of three particles with different x coordinates
particles = line.build_particles(zeta=0., delta=np.linspace(0,0.0015, 30),
                               x_norm=0, # in sigmas
                               px_norm=0, # in sigmas
                               nemitt_x=3e-6, nemitt_y=3e-6,
                               )

line.track(particles, num_turns=300, turn_by_turn_monitor=True)
z_sep, delta_sep = matcher.get_separatrix()
plt.figure(1)
plt.plot(line.record_last_track.zeta, line.record_last_track.delta, 'k.')
plt.plot(z_sep, delta_sep, 'r')
plt.plot(z_sep, -delta_sep, 'r')
plt.xlim(-3,3)
plt.xlabel('zeta [m]')
plt.ylabel('delta')

plt.show()