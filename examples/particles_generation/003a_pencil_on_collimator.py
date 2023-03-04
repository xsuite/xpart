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

# Load machine model
filename = ('../../../xtrack/test_data/hllhc15_noerrors_nobb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
line=xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])
line.build_tracker()

# Location of the collimator
at_element = 'tcp.d6l7.b1'
at_s = line.line.get_s_position(at_element) + 1.
y_cut = 3e-3 # position of the jaw
pencil_dr_sigmas = 3 # width of the pencil
side = '+' # side of the pencil

# Horizontal plane: generate gaussian distribution in normalized coordinates
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)

# Vertical plane: generate pencil beam in absolute coordinates
y_absolute, py_absolute = xp.generate_2D_pencil_with_absolute_cut(num_particles,
                    plane='y', absolute_cut=y_cut, dr_sigmas=pencil_dr_sigmas,
                    side=side, line=line,
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                    at_element=at_element, match_at_s=at_s)

# Longitudinal plane: generate gaussian distribution matched to bucket
zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian',
        sigma_z=10e-2, line=line)

# Combine the three planes
# (the normalized coordinates in y/py are blurred in order to preserve the geometric ones)
particles = line.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                y=y_absolute, py=py_absolute,
                x_norm=x_in_sigmas, px_norm=px_in_sigmas,
                zeta=zeta, delta=delta,
                at_element=at_element, match_at_s=at_s)

# Drift to at_s position for checking
drift_to_at_s = xt.Drift(length=at_s-line.get_s_position(at_element))
drift_to_at_s.track(particles)

# Checks and plots

tw_at_s = line.twiss(at_s=at_s)
norm_coords = tw_at_s.get_normalized_coordinates(
                particles, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                _force_at_element=0)

x_norm = norm_coords.x_norm
px_norm = norm_coords.px_norm
y_norm = norm_coords.y_norm
py_norm = norm_coords.py_norm


import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4, 7))
ax1 = fig1.add_subplot(3,2,1)
ax2 = fig1.add_subplot(3,2,3)
ax3 = fig1.add_subplot(3,2,5)
ax1.plot(x_norm, px_norm, '.', markersize=1)
ax1.set_xlabel(r'x [$\sigma$]')
ax1.set_ylabel(r'px [$\sigma$]')
ax1.set_xlim(np.array([-1, 1])*np.max(np.abs(x_norm))*1.1)
ax1.set_ylim(np.array([-1, 1])*np.max(np.abs(px_norm))*1.1)
ax2.plot(y_norm, py_norm, '.', markersize=1)
ax2.set_xlabel(r'y [$\sigma$]')
ax2.set_ylabel(r'py [$\sigma$]')
ax2.set_xlim(np.array([-1, 1])*np.max(np.abs(y_norm))*1.1)
ax2.set_ylim(np.array([-1, 1])*np.max(np.abs(py_norm))*1.1)
ax3.plot(zeta, delta*1000, '.', markersize=1)
ax3.set_xlabel(r'z [m]')
ax3.set_ylabel(r'$\delta$ [1e-3]')

ax21 = fig1.add_subplot(3,2,2)
ax22 = fig1.add_subplot(3,2,4)
ax23 = fig1.add_subplot(3,2,6)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax21.set_xlim(np.array([-1, 1])*np.max(np.abs(particles.x*1000))*1.1)
ax21.set_ylim(np.array([-1, 1])*np.max(np.abs(particles.px))*1.1)
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.axvline(x=y_cut*1000)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax22.set_xlim(np.array([-1, 1])*np.max(np.abs(particles.y*1000))*1.1)
ax22.set_ylim(np.array([-1, 1])*np.max(np.abs(particles.py))*1.1)
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33,
                     right=.96, wspace=.33)
plt.show()
