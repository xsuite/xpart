import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p
from scipy.constants import c as clight

import pymask as pm
import xpart as xp

sigma_z = 22.5e-2

rfbucket = xp.RFBucket(circumference=6911., gamma=27.6433,
                       mass=m_p, charge=qe, alpha_array=[0.003077672469],
                       harmonic_list=[4620], voltage_list=[3e6],
                       p_increment=0, phi_offset_list=[0.])

matcher = xp.RFBucketMatcher(rfbucket=rfbucket,
        distribution_type=xp.rfbucket_matching.ThermalDistribution,
        sigma_z=sigma_z)

n_part = int(1e6)

z_particles, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_part)

# Match transverse plane
line_file = ('../../xtrack/test_data/sps_w_spacecharge/'
             'optics_and_co_at_start_ring.json')

with open(line_file, 'r') as fid:
    ddd = json.load(fid)
RR = np.array(ddd['RR_madx'])
part_on_co = xp.Particles.from_dict(ddd['particle_on_madx_co'])

WW, WWinv, Rot = pm.compute_linear_normal_form(RR)

assert len(z_particles) == len(delta_particles)
num_particles = len(z_particles)

nemitt_x = 2e-6
nemitt_y = 2.5e-6

gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0

x_norm = np.sqrt(gemitt_x) * np.random.normal(size=num_particles)
px_norm = np.sqrt(gemitt_x) * np.random.normal(size=num_particles)
y_norm = np.sqrt(gemitt_y) * np.random.normal(size=num_particles)
py_norm = np.sqrt(gemitt_y) * np.random.normal(size=num_particles)

# Transform long. coordinates to normalized space
XX_norm = np.dot(WWinv, np.array([np.zeros(num_particles),
                                 np.zeros(num_particles),
                                 np.zeros(num_particles),
                                 np.zeros(num_particles),
                                 z_particles,
                                 delta_particles]))

XX_norm[0, :] = x_norm
XX_norm[1, :] = px_norm
XX_norm[2, :] = y_norm
XX_norm[3, :] = py_norm

# Transform to physical coordinates
XX = np.dot(WW, XX_norm)

part = part_on_co.copy()
part.x += XX[0, :]
part.px += XX[1, :]
part.y += XX[2, :]
part.py += XX[3, :]
part.zeta += XX[4, :]
part.delta += XX[5, :]
part.partid = np.arange(0, num_particles, dtype=np.int64)

y_rms = np.std(part.y)
py_rms = np.std(part.py)
x_rms = np.std(part.x)
px_rms = np.std(part.px)
delta_rms = np.std(part.delta)
zeta_rms = np.std(part.zeta)


assert np.isclose(zeta_rms, sigma_z, rtol=1e-3, atol=1e-15)
assert np.isclose(x_rms,
             np.sqrt(ddd['betx']*gemitt_x + ddd['dx']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)
assert np.isclose(y_rms,
             np.sqrt(ddd['bety']*gemitt_y + ddd['dy']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)

