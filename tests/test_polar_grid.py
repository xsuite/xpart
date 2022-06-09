# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp


def test_generate_2D_polar_grid():
    x_norm, px_norm, r_points, theta_points = xp.generate_2D_polar_grid(
            r_range=(0.1, 10), nr=8, theta_range=(0, np.pi/2), ntheta=10)

    assert len(x_norm) == len(px_norm) == len(r_points) == len(theta_points) == 80

    assert np.isclose(np.max(x_norm), 10)
    assert np.isclose(np.max(px_norm), 10)
    assert np.isclose(np.min(np.sqrt(x_norm**2 + px_norm**2)), 0.1)

    assert np.isclose(np.min(theta_points), 0)
    assert np.isclose(np.max(theta_points), np.pi/2)
    assert np.isclose(np.min(r_points), 0.1)
    assert np.isclose(np.max(r_points), 10)

