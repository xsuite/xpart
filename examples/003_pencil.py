import numpy as np

import xpart as xp

pos_cut_sigmas = 6.
dr_sigmas = 0.7
num_particles = 10000

def generate_2D_pencil(pos_cut_sigmas, dr_sigmas,
                       side='+'):

    assert side == '+' or side == '-' or side == '+-'

    r_min = np.abs(pos_cut_sigmas)
    r_max = r_min + dr_sigmas
    theta_max = np.arccos(pos_cut_sigmas/(np.abs(pos_cut_sigmas) + dr_sigmas))
    target_area = r_max**2/2*(2 * theta_max - np.sin(2 * theta_max))
    generated_area = (r_max**2 - r_min**2) * theta_max

    n_gen = int(num_particles*generated_area/target_area)

    x_gen, px_gen, r_points, theta_points = xp.generate_2D_uniform_circular_sector(
            # Should avoid regen most of the times
            num_particles=int(n_gen*1.5+100),
            r_range=(r_min, r_max),
            theta_range=(-theta_max, theta_max))

    mask = x_gen > r_min
    x_norm = x_gen[mask]
    px_norm = px_gen[mask]
    r_points = r_points[mask]
    theta_points = theta_points[mask]

    assert len(x_norm) > num_particles

    x_norm = x_norm[:num_particles]
    px_norm = px_norm[:num_particles]
    r_points = r_points[:num_particles]
    theta_points = theta_points[:num_particles]

    if side == '-':
        x_norm = -x_norm

    return x_norm, px_norm, r_points, theta_points

x_norm, px_norm, r_points, theta_points = generate_2D_pencil(
                                                 pos_cut_sigmas, dr_sigmas, side='-')
import matplotlib.pyplot as plt
plt.close('all')
#plt.plot(x_gen, px_gen, '.', markersize=1)
plt.plot(x_norm, px_norm, '.', markersize=1)
plt.axvline(x=pos_cut_sigmas)
#r_plot = np.linspace(0, 10, 10)
#plt.plot(r_plot * np.cos(theta_max), r_plot * np.sin(theta_max))
plt.xlabel(r'x [$\sigma$]')
plt.ylabel(r'px [$\sigma$]')
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.show()

