import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import pymask as pm
import xpart as xp

bunch_intensity = 1e11
sigma_z = 22.5e-2
n_part = int(5e6)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

line_file = ('../../xtrack/test_data/sps_w_spacecharge/'
             'optics_and_co_at_start_ring.json')
with open(line_file, 'r') as fid:
    ddd = json.load(fid)
RR = np.array(ddd['RR_madx'])
part_on_co = xp.Particles.from_dict(ddd['particle_on_madx_co'])

rfbucket = xp.RFBucket(circumference=6911., gamma=27.6433,
                       mass=m_p, charge_coulomb=qe, alpha_array=[0.003077672469],
                       harmonic_list=[4620], voltage_list=[3e6],
                       p_increment=0, phi_offset_list=[0.])

part = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_on_co=part_on_co, R_matrix=RR, rfbucket=rfbucket)

# CHECKS

y_rms = np.std(part.y)
py_rms = np.std(part.py)
x_rms = np.std(part.x)
px_rms = np.std(part.px)
delta_rms = np.std(part.delta)
zeta_rms = np.std(part.zeta)


gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
assert np.isclose(zeta_rms, sigma_z, rtol=1e-3, atol=1e-15)
assert np.isclose(x_rms,
             np.sqrt(ddd['betx']*gemitt_x + ddd['dx']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)
assert np.isclose(y_rms,
             np.sqrt(ddd['bety']*gemitt_y + ddd['dy']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)

