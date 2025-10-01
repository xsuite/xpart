# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np
from matplotlib import pyplot as plt
import xpart as xp
import xtrack as xt


def q_gaussian_1d(x, q, beta, normalize=False):
    """
    Args:
        x:
        q: q-parameter
        beta: beta for q-Gaussian
        normalize: if normalize area to 1

    Returns:
        q-Gaussian function defined on x

    """
    assert q < 5/3, "q must be less than 5/3"
    arg = 1 - (1 - q) * beta * x**2
    f = np.where(arg > 0, arg**(1 / (1 - q)), 0)
    if normalize:
        dx = x[1] - x[0]
        area = np.sum(f) * dx
        f /= area
    return f


bunch_intensity = 1e11
sigma_z = 22.5e-2
n_part = int(5e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

filename = ('../../../xtrack/test_data/sps_w_spacecharge'
            '/line_no_spacecharge_and_particle.json')
with open(filename, 'r') as fid:
    ddd = json.load(fid)
line = xt.Line.from_dict(ddd['line'])
line.particle_ref = xp.Particles.from_dict(ddd['particle'])

line.build_tracker()

q = 1.2
beta = 1

x_norm, px_norm, y_norm, py_norm = xp.generate_round_4D_q_gaussian_normalised(q=q, beta=beta, n_part=int(1e6))



# PLOT normalised x against 1D q-Gaussian
x = np.linspace(-10, 10, 1000)
f = q_gaussian_1d(x=x, q=q, beta=beta, normalize=True)
plt.plot(x, f, color='blue', label=f'1D q-Gaussian q={q}, beta={beta}')
plt.hist(x_norm, bins=200, density=True, label=f'sampled q-Gaussian q={q}, beta={beta}')
plt.legend()
plt.show()

particles = line.build_particles(
                               zeta=0, delta=1e-3,
                               x_norm=x_norm, # in sigmas
                               px_norm=px_norm, # in sigmas
                               y_norm=y_norm,
                               py_norm=py_norm,
                               nemitt_x=3e-6, nemitt_y=3e-6)

# CHECKS
y_rms = np.std(particles.y)
py_rms = np.std(particles.py)
x_rms = np.std(particles.x)
px_rms = np.std(particles.px)

print('y rms: ', y_rms, 'py rms: ', py_rms,'x rms: ', x_rms, 'px rms: ', px_rms)







