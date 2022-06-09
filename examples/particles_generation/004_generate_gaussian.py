# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import xpart as xp
import xtrack as xt

bunch_intensity = 1e11
sigma_z = 22.5e-2
n_part = int(5e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

filename = ('../../../xtrack/test_data/sps_w_spacecharge'
            '/line_no_spacecharge_and_particle.json')
with open(filename, 'r') as fid:
    ddd = json.load(fid)
tracker = xt.Tracker(line=xt.Line.from_dict(ddd['line']))
part_ref = xp.Particles.from_dict(ddd['particle'])


particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=part_ref,
         tracker=tracker)


#!end-doc-part

# CHECKS

y_rms = np.std(particles.y)
py_rms = np.std(particles.py)
x_rms = np.std(particles.x)
px_rms = np.std(particles.px)
delta_rms = np.std(particles.delta)
zeta_rms = np.std(particles.zeta)


fopt = ('../../../xtrack/test_data/sps_w_spacecharge/'
            'optics_and_co_at_start_ring.json')
with open(fopt, 'r') as fid:
    dopt = json.load(fid)
gemitt_x = nemitt_x/part_ref.beta0/part_ref.gamma0
gemitt_y = nemitt_y/part_ref.beta0/part_ref.gamma0
assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
assert np.isclose(x_rms,
             np.sqrt(dopt['betx']*gemitt_x + dopt['dx']**2*delta_rms**2),
             rtol=1e-2, atol=1e-15)
assert np.isclose(y_rms,
             np.sqrt(dopt['bety']*gemitt_y + dopt['dy']**2*delta_rms**2),
             rtol=1e-2, atol=1e-15)

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
