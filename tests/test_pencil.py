# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

TEST_DATA_FOLDER = pathlib.Path(__file__).parent / '../../xtrack/test_data'

from xobjects.test_helpers import for_all_test_contexts


def test_pencil_polar_coordinates():
    for side in ['+', '-', '+-']:
        _check_pencil_polar_coordinates(side)


def _check_pencil_polar_coordinates(side):
    x_norm, px_norm, r_points, theta_points = xp.generate_2D_pencil(
        num_particles=10000,
        pos_cut_sigmas=6.,
        dr_sigmas=0.7,
        side=side)

    if side == '+':
        assert np.all(x_norm > 6.)
    elif side == '-':
        assert np.all(x_norm < -6.)
    else:
        assert np.any(x_norm < -6.)
        assert np.any(x_norm > 6.)

    xo.assert_allclose(r_points, np.sqrt(x_norm**2 + px_norm**2),
                       rtol=1e-05, atol=1e-08)
    xo.assert_allclose(theta_points, np.arctan2(px_norm, x_norm),
                       rtol=1e-05, atol=1e-08)
    xo.assert_allclose(x_norm, r_points * np.cos(theta_points),
                       rtol=1e-05, atol=1e-08)
    xo.assert_allclose(px_norm, r_points * np.sin(theta_points),
                       rtol=1e-05, atol=1e-08)


@for_all_test_contexts
def test_pencil(test_context):
    num_particles = 10000
    nemitt_x = 2.5e-6
    nemitt_y = 3e-6

    # Load machine model (from pymask)
    filename = TEST_DATA_FOLDER.joinpath('lhc_no_bb/line_and_particle.json')
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
