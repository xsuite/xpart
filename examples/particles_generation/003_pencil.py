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
line=xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])
line.build_tracker()

# Horizontal plane: generate gaussian distribution in normalized coordinates
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)

# Vertical plane: generate pencil distribution in normalized coordinates
pencil_cut_sigmas = 6.
pencil_dr_sigmas = 0.7
y_in_sigmas, py_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=pencil_dr_sigmas,
                             side='+-')

# Longitudinal plane: generate gaussian distribution matched to bucket 
zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian',
        sigma_z=10e-2, line=line)

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
# tracker.track(particles, num_turns=10)

#!end-doc-part

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4, 7))
ax1 = fig1.add_subplot(3,2,1)
ax2 = fig1.add_subplot(3,2,3)
ax3 = fig1.add_subplot(3,2,5)
ax1.plot(x_in_sigmas, px_in_sigmas, '.', markersize=1)
ax1.set_xlabel(r'x [$\sigma$]')
ax1.set_ylabel(r'px [$\sigma$]')
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)
ax2.plot(y_in_sigmas, py_in_sigmas, '.', markersize=1)
ax2.set_xlabel(r'y [$\sigma$]')
ax2.set_ylabel(r'py [$\sigma$]')
ax2.axvline(x=pencil_cut_sigmas)
ax2.set_xlim(-7, 7)
ax2.set_ylim(-7, 7)
ax3.plot(zeta, delta*1000, '.', markersize=1)
ax3.set_xlabel(r'z [m]')
ax3.set_ylabel(r'$\delta$ [1e-3]')

ax21 = fig1.add_subplot(3,2,2)
ax22 = fig1.add_subplot(3,2,4)
ax23 = fig1.add_subplot(3,2,6)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33,
                     right=.96, wspace=.33)
plt.show()

