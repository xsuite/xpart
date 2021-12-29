import numpy as np

from .longitudinal import generate_longitudinal_coordinates
from .linear_normal_form import compute_linear_normal_form
from .build_particles import build_particles
from .particles import Particles


def generate_matched_gaussian_bunch(num_particles, total_intensity_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    particle_on_co=None,
                                    R_matrix=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    p_increment=0.,
                                    tracker=None,
                                    particle_ref=None,
                                    particle_class=Particles,
                                    co_search_settings=None,
                                    steps_r_matrix=None,
                                    _context=None, _buffer=None, _offset=None,
                                    ):

    if (particle_ref is not None and particle_on_co is not None):
        raise ValueError("`particle_ref` and `particle_on_co`"
                " cannot be provided at the same time")

    if particle_ref is None:
        assert particle_on_co is not None, (
            "`particle_ref` or `particle_on_co` must be provided!")
        particle_ref = particle_on_co

    zeta, delta = generate_longitudinal_coordinates(
            distribution='gaussian',
            num_particles=num_particles,
            particle_ref=(particle_ref if particle_ref is not None
                          else particle_on_co),
            tracker=tracker,
            circumference=circumference,
            momentum_compaction_factor=momentum_compaction_factor,
            rf_harmonic=rf_harmonic,
            rf_voltage=rf_voltage,
            rf_phase=rf_phase,
            p_increment=p_increment,
            sigma_z=sigma_z)

    assert len(zeta) == len(delta) == num_particles

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)


    part = build_particles(_context=_context, _buffer=_buffer, _offset=_offset,
                      R_matrix=R_matrix,
                      particle_class=particle_class,
                      particle_on_co=particle_on_co,
                      particle_ref=(
                          particle_ref if particle_on_co is  None else None),
                      tracker=tracker,
                      zeta=zeta, delta=delta,
                      x_norm=x_norm, px_norm=px_norm,
                      y_norm=y_norm, py_norm=py_norm,
                      scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                      weight=total_intensity_particles/num_particles,
                      co_search_settings=co_search_settings,
                      steps_r_matrix=steps_r_matrix)
    return part
