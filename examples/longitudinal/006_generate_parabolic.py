# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""
Simple example on how to generate a parabolic particle distribution
"""

import json
import xpart as xp
import xtrack as xt
import xobjects as xo

from xpart import parabolic_longitudinal_distribution
import matplotlib.pyplot as plt
import numpy as np

# Load the reference particle
filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.build_tracker()

# Specify the beam parameters
num_part = 1000000
sigma_z = 0.05

# Build a reference particle
p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                  delta=[10])

# Built a set of three particles with different x coordinates
particles = parabolic_longitudinal_distribution(
						num_particles=num_part,
						nemitt_x=3e-6, 
						nemitt_y=3e-6, 
						sigma_z=sigma_z,
						particle_ref=p0, 
						total_intensity_particles=1e10,
						line=line
						)
						
						
# Make a histogram
bin_heights, bin_borders = np.histogram(particles.zeta, bins=300)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2


# Generate the plots
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 7))
ax21 = fig1.add_subplot(3,1,1)
ax22 = fig1.add_subplot(3,1,2)
ax23 = fig1.add_subplot(3,1,3)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33, left=.18,
                     right=.96, wspace=.33)
                     
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(bin_centers, bin_heights, width=bin_widths)
ax2.set_ylabel('Counts')
ax2.set_xlabel(r'z [-]')
                     
plt.show()
