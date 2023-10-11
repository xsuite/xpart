# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import xpart as xp
import xtrack as xt

bunch_intensity = 1e11
sigma_dp = 2.3e-4
n_part = int(5e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

try:
    filename = ('../../../xtrack/test_data/sps_ions/line_and_particle.json')
    with open(filename, 'r') as fid:
        ddd = json.load(fid)
    line = xt.Line.from_dict(ddd)
except:
    # make something simpler
    line = xt.Line(
        elements=[xt.LineSegmentMap(
            qx=2.32, qy=4.87,
            dqx=0, dqy=0,
            length=25,
            betx=2,
            bety=3,
            dx=1)])
    line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, p0c=1e9)
line.build_tracker()

# force in all cases to generate a coasting beam
particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_dp=sigma_dp,
         coasting=True,
         line=line)

#!end-doc-part

# CHECKS

y_rms = np.std(particles.y)
py_rms = np.std(particles.py)
x_rms = np.std(particles.x)
px_rms = np.std(particles.px)
delta_rms = np.std(particles.delta)
zeta_rms = np.std(particles.zeta)



import matplotlib.pyplot as plt
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
plt.show()
