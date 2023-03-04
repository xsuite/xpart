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

scenario = 'ions' # can be "protons" or "ions"

if scenario == 'protons':
    filename = xt._pkg_root.parent.joinpath(
        'test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])
        line.particle_ref = xp.Particles.from_dict(input_data['particle'])
elif scenario == 'ions':
    # Load machine model (from pymask)
    filename = xt._pkg_root.parent.joinpath(
        'test_data/sps_ions/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
        line=xt.Line.from_dict(input_data)

line.build_tracker(_context=ctx)


rms_bunch_length=0.25
distribution = "gaussian"
n_particles = 100000
zeta, delta, matcher = xp.generate_longitudinal_coordinates(line=line,
                        num_particles=n_particles,
                        sigma_z=rms_bunch_length, distribution=distribution,
                        engine="single-rf-harmonic", return_matcher=True)
                        #engine="pyheadtail", return_matcher=True)

# Built a set of three particles with different x coordinates
particles = xp.build_particles(_context=ctx,
                               line=line,
                               zeta=zeta, delta=delta,
                               x_norm=0, # in sigmas
                               px_norm=0, # in sigmas
                               scale_with_transverse_norm_emitt=(3e-6, 3e-6)
                               )

zeta_sep, delta_sep = matcher.get_separatrix()
beta0 = line.line.particle_ref.beta0
plt.close('all')
plt.figure(1)
plt.hist2d(zeta, delta, bins=100,
        range=((-1.05 * np.max(zeta_sep), 1.05 * np.max(zeta_sep)),
               (-1.05*np.max(delta_sep), 1.05*np.max(delta_sep))),
        cmin=0.001)
plt.plot(zeta_sep, delta_sep, 'r')
plt.plot(zeta_sep, -delta_sep, 'r')
plt.xlabel('zeta [m]')
plt.ylabel('delta')

zeta_distr_y = matcher.tau_distr_y
zeta_distr_x = matcher.tau_distr_x * beta0
dx = zeta_distr_x[1] - zeta_distr_x[0]
hist, _  = np.histogram(zeta,
                range=(zeta_distr_x[0]-dx/2.,
                zeta_distr_x[-1]+dx/2.), bins=len(zeta_distr_x))
hist = hist / sum(hist) * sum(zeta_distr_y)
plt.figure(2)
plt.plot(zeta_distr_x, hist, 'bo', label="sampled line density")
plt.plot(zeta_distr_x, zeta_distr_y, 'r-', label="input line density")
plt.xlabel('ζ [m]')
plt.legend()

plt.show()