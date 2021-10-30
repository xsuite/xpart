import numpy as np

import xtrack as xt

from .longitudinal import generate_longitudinal_coordinates
from .linear_normal_form import compute_linear_normal_form
from .assemble_particles import assemble_particles


def generate_matched_gaussian_bunch(num_particles, total_intensity_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    particle_on_co, R_matrix,
                                    circumference,
                                    alpha_momentum_compaction,
                                    rf_harmonic,
                                    rf_voltage,
                                    rf_phase,
                                    p_increment=0.,
                                    particle_class=xt.Particles,
                                    _context=None, _buffer=None, _offset=None,
                                    ):

    zeta, delta = generate_longitudinal_coordinates(
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

    assert len(zeta) == len(delta) == num_particles

    gemitt_x = nemitt_x/particle_on_co.beta0/particle_on_co.gamma0
    gemitt_y = nemitt_y/particle_on_co.beta0/particle_on_co.gamma0

    x_norm = np.sqrt(gemitt_x) * np.random.normal(size=num_particles)
    px_norm = np.sqrt(gemitt_x) * np.random.normal(size=num_particles)
    y_norm = np.sqrt(gemitt_y) * np.random.normal(size=num_particles)
    py_norm = np.sqrt(gemitt_y) * np.random.normal(size=num_particles)


    part = assemble_particles(_context=_context, _buffer=_buffer, _offset=_offset,
                      R_matrix=R_matrix,
                      particle_class=particle_class,
                      particle_on_co=particle_on_co,
                      zeta=zeta, delta=delta,
                      x_norm=x_norm, px_norm=px_norm,
                      y_norm=y_norm, py_norm=py_norm,
                      weight=total_intensity_particles/num_particles)
    return part
