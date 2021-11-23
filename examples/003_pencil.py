import numpy as np

import xpart as xp

pos_cut_sigmas = 6.
dr_sigmas = 0.7
num_particles = 10000

x_norm, px_norm, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pos_cut_sigmas, dr_sigmas=dr_sigmas,
                             side='+-')

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

