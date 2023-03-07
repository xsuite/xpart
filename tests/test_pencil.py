# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_pencil(test_context):
    num_particles = 10000
    nemitt_x = 2.5e-6
    nemitt_y = 3e-6

    # Load machine model (from pymask)
    filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.build_tracker()
    particle_sample = xp.Particles.from_dict(input_data['particle'])

    # Horizontal plane: generate gaussian distribution in normalized coordinates
    x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)

    # Vertical plane: generate pencil distribution in normalized coordinates
    pencil_cut_sigmas = 6.
    pencil_dr_sigmas = 0.7
    y_in_sigmas, py_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                                 num_particles=num_particles,
                                 pos_cut_sigmas=pencil_cut_sigmas,
                                 dr_sigmas=pencil_dr_sigmas,
                                 side='+-')

    # Longitudinal plane: generate gaussian distribution matched to bucket
    zeta, delta = xp.generate_longitudinal_coordinates(
            num_particles=num_particles, distribution='gaussian',
            sigma_z=10e-2, particle_ref=particle_sample, line=line)

    # Build particles:
    #    - scale with given emittances
    #    - transform to physical coordinates (using 1-turn matrix)
    #    - handle dispersion
    #    - center around the closed orbit
    particles = xp.build_particles(_context=test_context,
                                   line=line,
                                   particle_ref=particle_sample,
                                   zeta=zeta, delta=delta,
                                   x_norm=x_in_sigmas, px_norm=px_in_sigmas,
                                   y_norm=y_in_sigmas, py_norm=py_in_sigmas,
                                   nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    dct = particles.to_dict() # transfers it to cpu
    assert np.min(np.abs(dct['y'])) > 0.0018
    assert np.max(np.abs(dct['y'])) < 0.0021
