# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp

def test_generate_2D_uniform_circular_sector():

    (x_norm, px_norm, r_points, theta_points
            ) = xp.generate_2D_uniform_circular_sector(
            num_particles=10000, r_range=(0.6, 0.9),
            theta_range=(0.25*np.pi, 1.75*np.pi))

    assert (len(x_norm) == len(px_norm) == len(r_points)
            == len(theta_points) == 10000)

    xo.assert_allclose(np.max(np.abs(x_norm)), 0.9, rtol=1e-2, atol=1e-08)
    xo.assert_allclose(np.max(px_norm), 0.9, rtol=1e-2, atol=1e-08)
    xo.assert_allclose(np.min(np.sqrt(x_norm**2 + px_norm**2)), 0.6,
                       rtol=1e-2, atol=1e-08)

    xo.assert_allclose(np.min(theta_points), 0.25*np.pi,
                       rtol=1e-2, atol=1e-08)
    xo.assert_allclose(np.max(theta_points), 1.75*np.pi,
                       rtol=1e-2, atol=1e-08)
    xo.assert_allclose(np.min(r_points), 0.6, rtol=1e-2, atol=1e-08)
    xo.assert_allclose(np.max(r_points), 0.9, rtol=1e-2, atol=1e-08)
