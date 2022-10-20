# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo


import xpart as xp
import xtrack as xt
import time

ctx = xo.ContextCpu()

# Load machine model
filename = xt._pkg_root.parent.joinpath('test_data/sps_ions/line.json')
#filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data))
#tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']))

rms_bunch_length=0.14
distribution = "gaussian"
n_particles = 10000
zeta, delta, matcher = xp.generate_longitudinal_coordinates(tracker=tracker,
                                                 num_particles=n_particles,
                                                 sigma_z=rms_bunch_length, distribution=distribution,
                                                 engine="single-rf-harmonic", return_matcher=True)
                                                 #engine="pyheadtail", return_matcher=True)

# Built a set of three particles with different x coordinates
particles = xp.build_particles(_context=ctx,
                               tracker=tracker,
                               zeta=0., delta=np.linspace(0,0.0015, 30),
                               x_norm=0, # in sigmas
                               px_norm=0, # in sigmas
                               nemitt_x=3e-6, nemitt_y=3e-6,
                               )

tracker.track(particles, num_turns=300, turn_by_turn_monitor=True)
z_sep, delta_sep = matcher.get_separatrix()
plt.figure(1)
plt.plot(tracker.record_last_track.zeta, tracker.record_last_track.delta, 'k.')
plt.plot(z_sep, delta_sep, 'r')
plt.plot(z_sep, -delta_sep, 'r')
plt.xlim(-3,3)
plt.xlabel('zeta [m]')
plt.ylabel('delta')

plt.show()