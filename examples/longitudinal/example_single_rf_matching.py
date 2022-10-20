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

scenario = 'protons' # can be "protons" or "ions"

if scenario == 'protons':
    filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])
        line.particle_ref = xp.Particles.from_dict(input_data['particle'])
elif scenario == 'ions':
    # Load machine model (from pymask)
    filename = xt._pkg_root.parent.joinpath('test_data/sps_ions/line.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
        line=xt.Line.from_dict(input_data)

tracker = xt.Tracker(_context=ctx, line=line)

rms_bunch_length=0.16
distribution = "gaussian"
n_particles = 100000
zeta, delta, matcher = xp.generate_longitudinal_coordinates(tracker=tracker,
                                                 num_particles=n_particles,
                                                 sigma_z=rms_bunch_length, distribution=distribution,
                                                 engine="single-rf-harmonic", return_matcher=True)
                                                 #engine="pyheadtail", return_matcher=True)

# Built a set of three particles with different x coordinates
particles = xp.build_particles(_context=ctx,
                               tracker=tracker,
                               zeta=zeta, delta=delta,
                               x_norm=0, # in sigmas
                               px_norm=0, # in sigmas
                               scale_with_transverse_norm_emitt=(3e-6, 3e-6)
                               )

x_sep, y_sep = matcher.get_separatrix()
tau = zeta/tracker.line.particle_ref.beta0
plt.close('all')
plt.figure(1)
plt.hist2d(zeta, delta, bins=100, range=((-0.7,0.7), (-0.001, 0.001)), cmin=0.001)
plt.plot(x_sep, y_sep, 'r')
plt.plot(x_sep, -y_sep, 'r')
plt.xlabel('zeta [m]')
plt.ylabel('delta')

tau_distr_y = matcher.tau_distr_y
tau_distr_x = matcher.tau_distr_x
dx = tau_distr_x[1] - tau_distr_x[0]
hist, _  = np.histogram(tau, range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.), bins=len(tau_distr_x))
hist = hist / sum(hist) * sum(tau_distr_y)
plt.figure(2)
plt.plot(tau_distr_x, hist, 'bo', label="sampled line density")
plt.plot(tau_distr_x, tau_distr_y, 'r-', label="input line density")
plt.xlabel('ζ [m]')
plt.legend()

plt.show()