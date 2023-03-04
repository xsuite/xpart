# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xpart as xp
import xtrack as xt

num_particles = 10000
nemitt_x = 2.5e-6
nemitt_y = 3e-6

# Load machine model (from pymask)
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])
line.build_tracker()

# Horizontal plane: generate cut halo distribution
(x_in_sigmas, px_in_sigmas, r_points, theta_points
    )= xp.generate_2D_uniform_circular_sector(
                                          num_particles=num_particles,
                                          r_range=(0.6, 0.9), # sigmas
                                          theta_range=(0.25*np.pi, 1.75*np.pi))

# Vertical plane: all particles on the closed orbit
y_in_sigmas = 0.
py_in_sigmas = 0.

# Longitudinal plane: all particles off momentum by 1e-3
zeta = 0.
delta = 1e-3

# Build particles:
#    - scale with given emittances
#    - transform to physical coordinates (using 1-turn matrix)
#    - handle dispersion
#    - center around the closed orbit
particles = line.build_particles(
            zeta=zeta, delta=delta,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

# Absolute coordinates can be inspected in particle.x, particles.px, etc.

# Tracking can be done with:
# line.track(particles, num_turns=10)

#!end-doc-part

assert (len(x_in_sigmas) == len(px_in_sigmas)
        == len(r_points) == len(theta_points) == 10000)

assert np.isclose(np.max(np.abs(x_in_sigmas)), 0.9, rtol=1e-2)
assert np.isclose(np.max(px_in_sigmas), 0.9, rtol=1e-2)
assert np.isclose(np.min(np.sqrt(x_in_sigmas**2 + px_in_sigmas**2)), 0.6, rtol=1e-2)

assert np.isclose(np.min(theta_points), 0.25*np.pi, rtol=1e-2)
assert np.isclose(np.max(theta_points), 1.75*np.pi, rtol=1e-2)
assert np.isclose(np.min(r_points), 0.6, rtol=1e-2)
assert np.isclose(np.max(r_points), 0.9, rtol=1e-2)

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4, 7))
ax1 = fig1.add_subplot(3,2,1)
ax2 = fig1.add_subplot(3,2,3)
ax3 = fig1.add_subplot(3,2,5)
ax1.plot(x_in_sigmas, px_in_sigmas, '.', markersize=1)
ax1.set_xlabel(r'x [$\sigma$]')
ax1.set_ylabel(r'px [$\sigma$]')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax2.plot(y_in_sigmas, py_in_sigmas, '.', markersize=1)
ax2.set_xlabel(r'y [$\sigma$]')
ax2.set_ylabel(r'py [$\sigma$]')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax3.plot(zeta, delta*1000, '.', markersize=1)
ax3.set_xlabel(r'z [m]')
ax3.set_ylabel(r'$\delta$ [1e-3]')
ax3.set_xlim(-0.05, 0.05)
ax3.set_ylim(-1.5, 1.5)

ax21 = fig1.add_subplot(3,2,2)
ax22 = fig1.add_subplot(3,2,4)
ax23 = fig1.add_subplot(3,2,6)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax22.set_xlim(-1, 1)
ax22.set_ylim(-5e-6, 5e-6)
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
ax23.set_xlim(-0.05, 0.05)
ax23.set_ylim(-1.5, 1.5)
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33,
                     right=.96, wspace=.33)
plt.show()
