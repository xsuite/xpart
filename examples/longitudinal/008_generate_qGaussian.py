# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""
Simple example on how to generate a qGaussian particle distribution
"""

import json
import xpart as xp
import xtrack as xt
import xobjects as xo

from xpart import build_particles
from xpart.longitudinal import generate_qgaussian_longitudinal_coordinates

import matplotlib.pyplot as plt
import numpy as np

# Decide context
test_on_gpu = False
if test_on_gpu:
    context = xo.ContextCupy()
else:
    context = xo.ContextCpu(omp_num_threads='auto')

# Load the reference particle
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.build_tracker(_context=context)

# Specify the beam parameters
q = 1.3  # example values with fatter tails
total_intensity_particles = 1e10
num_particles = 1000000
sigma_z = 0.05
nemitt_x = 3e-6
nemitt_y = 3e-6
# Build a reference particle
p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                  delta=[10])

# Build Guassian normalized coordinates
x_norm = np.random.normal(size=num_particles)
px_norm = np.random.normal(size=num_particles)
y_norm = np.random.normal(size=num_particles)
py_norm = np.random.normal(size=num_particles)


# Generate parabolic coordinates
zeta, delta = generate_qgaussian_longitudinal_coordinates(num_particles=num_particles,
                                                     	 nemitt_x=nemitt_x, 
                                                     	 nemitt_y=nemitt_y, 
                                                     	 sigma_z=sigma_z,
                                                     	 particle_ref=p0,
                                                     	 line=line, q=q)
				
# Build particle object
particles = build_particles(_context=context, particle_ref=p0,
			zeta=zeta, delta=delta, 
			x_norm=x_norm, px_norm=px_norm,
			y_norm=y_norm, py_norm=py_norm,
			nemitt_x=nemitt_x, nemitt_y=nemitt_y,
			weight=total_intensity_particles/num_particles, line=line)		
						
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
