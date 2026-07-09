# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp


def test_generate_2D_polar_grid():
    x_norm, px_norm, r_points, theta_points = xp.generate_2D_polar_grid(
            r_range=(0.1, 10), nr=8, theta_range=(0, np.pi/2), ntheta=10)

    assert len(x_norm) == len(px_norm) == len(r_points) == len(theta_points) == 80

    xo.assert_allclose(np.max(x_norm), 10, rtol=1e-05, atol=1e-08)
    xo.assert_allclose(np.max(px_norm), 10, rtol=1e-05, atol=1e-08)
    xo.assert_allclose(
        np.min(np.sqrt(x_norm**2 + px_norm**2)), 0.1,
        rtol=1e-05, atol=1e-08)

    xo.assert_allclose(np.min(theta_points), 0, rtol=1e-05, atol=1e-08)
    xo.assert_allclose(np.max(theta_points), np.pi/2, rtol=1e-05, atol=1e-08)
    xo.assert_allclose(np.min(r_points), 0.1, rtol=1e-05, atol=1e-08)
    xo.assert_allclose(np.max(r_points), 10, rtol=1e-05, atol=1e-08)
