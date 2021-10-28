import numpy as np

from .longitudinal import generate_longitudinal_coordinates


try:
    import pymask as pm # TODO: Temporary...
except Exception:
    print('pymask not available')

def generate_matched_gaussian_bunch(num_particles, total_intensity_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    particle_on_co, R_matrix,
                                    circumference,
                                    alpha_momentum_compaction,
                                    rf_harmonic,
                                    rf_voltage,
                                    rf_phase,
                                    p_increment=0.
                                    ):

    z_particles, delta_particles = generate_longitudinal_coordinates(
            distribution='gaussian',
            mass0=particle_on_co.mass0,
            q0=particle_on_co.q0,
            gamma0=particle_on_co.gamma0,
            num_particles=num_particles,
            circumference=circumference,
            alpha_momentum_compaction=alpha_momentum_compaction,
            rf_harmonic=rf_harmonic,
            rf_voltage=rf_voltage,
            rf_phase=rf_phase,
            p_increment=p_increment,
            sigma_z=sigma_z)

    WW, WWinv, Rot = pm.compute_linear_normal_form(R_matrix)

    assert len(z_particles) == len(delta_particles) == num_particles

    gemitt_x = nemitt_x/particle_on_co.beta0/particle_on_co.gamma0
    gemitt_y = nemitt_y/particle_on_co.beta0/particle_on_co.gamma0

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

    part = particle_on_co.copy()
    part.x += XX[0, :]
    part.px += XX[1, :]
    part.y += XX[2, :]
    part.py += XX[3, :]
    part.zeta += XX[4, :]
    part.delta += XX[5, :]
    part.particle_id = np.arange(0, num_particles, dtype=np.int64)
    part.weight = total_intensity_particles/num_particles

    return part
