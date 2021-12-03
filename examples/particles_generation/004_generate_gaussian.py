import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import xpart as xp
import xtrack as xt

bunch_intensity = 1e11
sigma_z = 22.5e-2
n_part = int(5e6)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

filename = ('../../../xtrack/test_data/sps_w_spacecharge'
            '/line_no_spacecharge_and_particle.json')
with open(filename, 'r') as fid:
    ddd = json.load(fid)
tracker = xt.Tracker(line=xt.Line.from_dict(ddd['line']))
part_ref = xp.Particles.from_dict(ddd['particle'])


part = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=part_ref,
         tracker=tracker)


#!end-doc-part

# CHECKS

y_rms = np.std(part.y)
py_rms = np.std(part.py)
x_rms = np.std(part.x)
px_rms = np.std(part.px)
delta_rms = np.std(part.delta)
zeta_rms = np.std(part.zeta)


fopt = ('../../../xtrack/test_data/sps_w_spacecharge/'
            'optics_and_co_at_start_ring.json')
with open(fopt, 'r') as fid:
    dopt = json.load(fid)
gemitt_x = nemitt_x/part_ref.beta0/part_ref.gamma0
gemitt_y = nemitt_y/part_ref.beta0/part_ref.gamma0
assert np.isclose(zeta_rms, sigma_z, rtol=1e-3, atol=1e-15)
assert np.isclose(x_rms,
             np.sqrt(dopt['betx']*gemitt_x + dopt['dx']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)
assert np.isclose(y_rms,
             np.sqrt(dopt['bety']*gemitt_y + dopt['dy']**2*delta_rms**2),
             rtol=1e-3, atol=1e-15)

