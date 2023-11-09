# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .general import _print

from .longitudinal import generate_longitudinal_coordinates
from .build_particles import build_particles

import xpart as xp # To get the right Particles class depending on pyheatail interface state

def generate_matched_gaussian_bunch(num_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    total_intensity_particles=None,
                                    particle_on_co=None,
                                    R_matrix=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    p_increment=0.,
                                    tracker=None,
                                    line=None,
                                    particle_ref=None,
                                    particles_class=None,
                                    engine=None,
                                    _context=None, _buffer=None, _offset=None,
                                    **kwargs, # They are passed to build_particles
                                    ):

    '''
    Generate a matched Gaussian bunch.

    Parameters
    ----------
    line : xpart.Line
        Line for which the bunch is generated.
    num_particles : int
        Number of particles to be generated.
    nemitt_x : float
        Normalized emittance in the horizontal plane (in m rad).
    nemitt_y : float
        Normalized emittance in the vertical plane (in m rad).
    sigma_z : float
        RMS bunch length in meters.
    total_intensity_particles : float
        Total intensity of the bunch in particles.

    Returns
    -------
    part : xpart.Particles
        Particles object containing the generated particles.

    '''

    if line is not None and tracker is not None:
        raise ValueError(
            'line and tracker cannot be provided at the same time.')

    if tracker is not None:
        _print(
            "The argument tracker is deprecated. Please use line instead.",
            DeprecationWarning)
        line = tracker.line

    if line is not None:
        assert line.tracker is not None, ("The line has no tracker. Please use "
                                          "`Line.build_tracker()`")

    if (particle_ref is not None and particle_on_co is not None):
        raise ValueError("`particle_ref` and `particle_on_co`"
                " cannot be provided at the same time")

    if particle_ref is None:
        if particle_on_co is not None:
            particle_ref = particle_on_co
        elif line is not None and line.particle_ref is not None:
            particle_ref = line.particle_ref
        else:
            raise ValueError(
                "`line`, `particle_ref` or `particle_on_co` must be provided!")

    zeta, delta = generate_longitudinal_coordinates(
            distribution='gaussian',
            num_particles=num_particles,
            particle_ref=(particle_ref if particle_ref is not None
                          else particle_on_co),
            line=line,
            circumference=circumference,
            momentum_compaction_factor=momentum_compaction_factor,
            rf_harmonic=rf_harmonic,
            rf_voltage=rf_voltage,
            rf_phase=rf_phase,
            p_increment=p_increment,
            sigma_z=sigma_z,
            engine=engine,
            **kwargs)

    assert len(zeta) == len(delta) == num_particles

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)

    if total_intensity_particles is None:
        # go to particles.weight = 1
        total_intensity_particles = num_particles


    part = build_particles(_context=_context, _buffer=_buffer, _offset=_offset,
                      R_matrix=R_matrix,
                      particles_class=particles_class,
                      particle_on_co=particle_on_co,
                      particle_ref=(
                          particle_ref if particle_on_co is  None else None),
                      line=line,
                      zeta=zeta, delta=delta,
                      x_norm=x_norm, px_norm=px_norm,
                      y_norm=y_norm, py_norm=py_norm,
                      nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                      weight=total_intensity_particles/num_particles,
                      **kwargs)
    return part


def generate_matched_gaussian_beam(filling_scheme,
                                   num_particles,
                                   nemitt_x, nemitt_y, sigma_z,
                                   total_intensity_particles=None,
                                   particle_on_co=None,
                                   R_matrix=None,
                                   circumference=None,
                                   momentum_compaction_factor=None,
                                   rf_harmonic=None,
                                   rf_voltage=None,
                                   rf_phase=None,
                                   p_increment=0.,
                                   tracker=None,
                                   line=None,
                                   particle_ref=None,
                                   particles_class=None,
                                   engine=None,
                                   _context=None, _buffer=None, _offset=None,
                                   **kwargs, # They are passed to build_particles
                                   ):

    bunches_for_this_processor = filling_scheme.get_bunches()

    n_levels = int(np.ceil(np.log(len(bunches_for_this_processor)) / np.log(2))) + 1
    bunches_tree = []
    for i in range(n_levels):
        bunches_tree.append([])

    # the bunches are generated and summed using a binary tree approach which scales better than a simple sum in the
    # case in which there are many bunches in one processor

    macro_bunch = generate_matched_gaussian_bunch(num_particles=num_particles*len(filling_scheme.get_bunches()),
                                                  nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                                                  total_intensity_particles=total_intensity_particles,
                                                  particle_on_co=particle_on_co,
                                                  R_matrix=R_matrix,
                                                  circumference=circumference,
                                                  momentum_compaction_factor=momentum_compaction_factor,
                                                  rf_harmonic=rf_harmonic,
                                                  rf_voltage=rf_voltage,
                                                  rf_phase=rf_phase,
                                                  p_increment=p_increment,
                                                  tracker=tracker,
                                                  line=line,
                                                  particle_ref=particle_ref,
                                                  particles_class=particles_class,
                                                  engine=engine,
                                                  _context=_context, _buffer=_buffer, _offset=_offset,
                                                  **kwargs,  # They are passed to build_particles
                                                  )

    bunch_spacing = filling_scheme.bunch_spacing
    bunch_id_bucket_id_map = filling_scheme.bunch_id_bucket_id_map
    for count, b_id in enumerate(filling_scheme.get_bunches()):
        macro_bunch.zeta[count*num_particles: (count + 1)*num_particles] += bunch_spacing*bunch_id_bucket_id_map[b_id]

    return macro_bunch
