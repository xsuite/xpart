import numpy as np

import xpart as xp

pos_cut_sigmas = 6.
dr_sigmas = 1.
num_particles = 10000

r_min = np.abs(pos_cut_sigmas)
r_max = r_min + dr_sigmas
theta_max = np.arccos(pos_cut_sigmas/(np.abs(pos_cut_sigmas) + dr_sigmas))
target_area = r_max**2/2*(2 * theta_max - np.sin(2 * theta_max))
generated_area = (r_max**2 - r_min**2) * theta_max

n_gen = int(num_particles*generated_area/target_area)

x_norm, px_norm, r_points, theta_points = xp.generate_2D_uniform_circular_sector(
        # Should avoid regen most of the times
        num_particles=int(n_gen*1.5+100),
        r_range=(r_min, r_max),
        theta_range=(-theta_max, theta_max))


mask = x_norm > r_min
x_norm = x_norm[mask]
px_norm = px_norm[mask]
r_points = r_points[mask]
theta_points = theta_points[mask]

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(x_norm, px_norm, '.', markersize=1)
plt.plot(x_norm[mask], px_norm[mask], '.', markersize=1)
plt.axvline(x=pos_cut_sigmas)
r_plot = np.linspace(0, 10, 10)
plt.plot(r_plot * np.cos(theta_max), r_plot * np.sin(theta_max))
plt.show()

