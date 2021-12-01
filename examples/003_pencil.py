import json
import numpy as np

import xpart as xp
import xtrack as xt

num_particles = 10000
nemitt_x = 2.5e-6
nemitt_y = 3e-6

# Load machine model (from pymask)
filename = ('../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
tracker = xt.Tracker(line=xt.Line.from_dict(input_data['line']))
particle_sample = xp.Particles.from_dict(input_data['particle'])


# Vertical pencil
pencil_cut_sigmas = 6.
pencil_dr_sigmas = 0.7
y_in_sigmas, py_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=pencil_dr_sigmas,
                             side='+-')

# Horizontal gaussian
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)

# Longitudinal - matched to bucket 
zeta, delta = xp.longitudinal.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian',
        sigma_z=10e-2, particle_ref=particle_sample, tracker=tracker)

# Build particles:
#    - scale with given emittances
#    - transform to physical coordinates (using 1-turn matrix)
#    - handle dispersion
#    - center around the closed orbit
particles = xp.build_particles(
            tracker=tracker,
            particle_ref=particle_sample,
            zeta=zeta, delta=delta,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y))


import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
#plt.plot(x_gen, px_gen, '.', markersize=1)
ax1.plot(y_in_sigmas, py_in_sigmas, '.', markersize=1)
ax1.axvline(x=pencil_cut_sigmas)
#r_plot = np.linspace(0, 10, 10)
#plt.plot(r_plot * np.cos(theta_max), r_plot * np.sin(theta_max))
ax1.set_xlabel(r'x [$\sigma$]')
ax1.set_ylabel(r'px [$\sigma$]')
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)
ax2.plot(x_in_sigmas, px_in_sigmas, '.', markersize=1)
plt.show()

