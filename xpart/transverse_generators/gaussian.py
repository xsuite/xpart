# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

def generate_2D_gaussian(num_particles):

    '''
    Generate a 2D Gaussian distribution.

    Parameters
    ----------
    num_particles : int
        Number of particles to be generated.

    Returns
    -------
    x1 : np.ndarray
        First normalized coordinate.
    x2 : np.ndarray
        Second normalized coordinate.

    '''

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)

    return x_norm, px_norm

