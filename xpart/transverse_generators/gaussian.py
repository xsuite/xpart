# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

def generate_2D_gaussian(num_particles):

    """
    Generate a 2D Gaussian distribution in normalized coordinates.

    Both returned coordinates are sampled independently from a standard normal
    distribution.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.

    Returns
    -------
    x_norm : np.ndarray
        First normalized coordinate.
    px_norm : np.ndarray
        Second normalized coordinate.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp

        np.random.seed(12345)

        x_norm, px_norm = xp.generate_2D_gaussian(num_particles=4)

        x_norm  # [-0.204708, 0.478943, -0.519439, -0.55573]
        px_norm # [1.965781, 1.393406, 0.092908, 0.281746]
    """

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)

    return x_norm, px_norm
