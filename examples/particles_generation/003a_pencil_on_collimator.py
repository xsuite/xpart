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
tracker = xt.Tracker(line=xt.Line.from_dict(input_data['line']))
tracker.particle_ref = xp.Particles.from_dict(input_data['particle'])

# Location of the collimator
at_element = 'tcp.d6l7.b1'
at_s = tracker.line.get_s_position(at_element) + 1.
y_cut = 3e-3 # position of the jaw
pencil_dr_sigmas = 3 # width of the pencil

tw_at_s = tracker.twiss(at_s=at_s)
drift_to_at_s = xt.Drift(length=at_s - tracker.line.get_s_position(at_element))

# I generate a particle exactly on the jaw with no normalized py
p_on_cut_at_element = tracker.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                   y=y_cut,
                                   x_norm=0, px_norm=0, py_norm=0,
                                   zeta_norm=0, pzeta_norm=0,
                                   at_element=at_element, match_at_s=at_s)
p_on_cut_at_s = p_on_cut_at_element.copy()
drift_to_at_s.track(p_on_cut_at_s)

# Get cut in (accurate) sigmas
p_on_cut_norm = tw_at_s.get_normalized_coordinates(p_on_cut_at_s,
                                        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        _force_at_element=0 # the twiss has only this element
                                        )
pencil_cut_sigmas = p_on_cut_norm.y_norm


# Generate pencil in y_norm - py_norm plane only
y_in_sigmas, py_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=pencil_dr_sigmas,
                             side='+')

# Generate geometric coordinates in un y/py plane only
# (by construction y_cut is preserved)
p_pencil_y_only_at_element = tracker.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                    y_norm=y_in_sigmas, py_norm=py_in_sigmas,
                                    zeta_norm=0, pzeta_norm=0,
                                    x_norm=0, px_norm=0,
                                    at_element=at_element, match_at_s=at_s)
p_pencil_y_only_at_s = p_pencil_y_only_at_element.copy()
drift_to_at_s.track(p_pencil_y_only_at_s)

# Add other coordinates without perturbing the y/py plane
# (the normalized coordinates are blurred in order to preserve the geometric ones)

# Horizontal plane: generate gaussian distribution in normalized coordinates
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)

# Longitudinal plane: generate gaussian distribution matched to bucket
zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian',
        sigma_z=10e-2, tracker=tracker)

particles = tracker.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                y=p_pencil_y_only_at_s.y, py=p_pencil_y_only_at_s.py,
                x_norm=x_in_sigmas, px_norm=px_in_sigmas,
                zeta=zeta, delta=delta,
                at_element=at_element, match_at_s=at_s)

# Drift to match position for checking

drift_to_at_s.track(particles)

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
ax22.axvline(x=y_cut*1000)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33,
                     right=.96, wspace=.33)
plt.show()
