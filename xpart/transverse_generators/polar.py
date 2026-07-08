# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import itertools

import numpy as np

def _configure_grid(vname, v_grid, dv, v_range, nv):

    # Check input consistency
    if v_grid is not None:
        assert dv is None, (f'd{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert nv is None, (f'n{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert v_range is None, (f'{vname}_range cannot be given '
                                 f'if {vname}_grid is provided')
        ddd = np.diff(v_grid)
        assert np.allclose(ddd,ddd[0]), (f'{vname}_grid must be '
                                          'unifirmly spaced')
    else:
        assert v_range is not None, (f'{vname}_grid or {vname}_range '
                                     f'must be provided')
        assert len(v_range)==2, (f'{vname}_range must be in the form '
                                 f'({vname}_min, {vname}_max)')
        if dv is not None:
            assert nv is None, (f'n{vname} cannot be given '
                                    f'if d{vname} is provided ')
            v_grid = np.arange(v_range[0], v_range[1]+0.1*dv, dv)
        else:
            assert nv is not None, (f'n{vname} must be given '
                                    f'if d{vname} is not provided ')
            v_grid = np.linspace(v_range[0], v_range[1], nv)

    return v_grid


def generate_2D_polar_grid(
        r_range=None, r_grid=None, dr=None, nr=None,
        theta_range=None, theta_grid=None, dtheta=None, ntheta=None):

    """
    Generate points on a 2D polar grid.

    The radial and angular grids can be provided explicitly with `r_grid` and
    `theta_grid`, or built from ranges. For each coordinate, provide either an
    explicit grid, or a range together with either a step size or a number of
    points. The returned arrays are flattened over all `(r, theta)`
    combinations.

    Parameters
    ----------
    r_range : tuple of float, optional
        Radial range `(r_min, r_max)`. Required if `r_grid` is not provided.
    r_grid : array_like, optional
        Explicit uniformly spaced radial grid. If provided, `r_range`, `dr`,
        and `nr` must not be provided.
    dr : float, optional
        Radial step used with `r_range`. Cannot be provided together with `nr`.
    nr : int, optional
        Number of radial points used with `r_range`. Required when `r_grid` and
        `dr` are not provided.
    theta_range : tuple of float, optional
        Angular range `(theta_min, theta_max)` in rad. Required if
        `theta_grid` is not provided.
    theta_grid : array_like, optional
        Explicit uniformly spaced angular grid in rad. If provided,
        `theta_range`, `dtheta`, and `ntheta` must not be provided.
    dtheta : float, optional
        Angular step in rad used with `theta_range`. Cannot be provided
        together with `ntheta`.
    ntheta : int, optional
        Number of angular points used with `theta_range`. Required when
        `theta_grid` and `dtheta` are not provided.

    Returns
    -------
    a1 : np.ndarray
        First Cartesian normalized coordinate, equal to
        `r_all * cos(theta_all)`.
    a2 : np.ndarray
        Second Cartesian normalized coordinate, equal to
        `r_all * sin(theta_all)`.
    r_all : np.ndarray
        Radial coordinate for each generated point.
    theta_all : np.ndarray
        Angular coordinate in rad for each generated point.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp

        a1, a2, r, theta = xp.generate_2D_polar_grid(
            r_range=(1.0, 2.0), nr=2,
            theta_range=(0.0, np.pi / 2), ntheta=3)

        a1     # [1.0, 0.707107, 0.0, 2.0, 1.414214, 0.0]
        a2     # [0.0, 0.707107, 1.0, 0.0, 1.414214, 2.0]
        r      # [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        theta  # [0.0, 0.785398, 1.570796, 0.0, 0.785398, 1.570796]
    """

    _r_grid = _configure_grid('r', r_grid, dr, r_range, nr)
    _theta_grid = _configure_grid('theta', theta_grid, dtheta,
                                  theta_range, ntheta)

    temp = np.array([(ii[0], ii[1]) for ii in
                    itertools.product(_r_grid, _theta_grid)])

    r_all = temp[:, 0]
    theta_all = temp[:, 1]

    a1 = r_all*np.cos(theta_all)
    a2 = r_all*np.sin(theta_all)

    return a1, a2, r_all, theta_all

def generate_2D_uniform_circular_sector(num_particles, r_range=(0, 1),
                                        theta_range=(0, 2*np.pi)):

    """
    Generate points uniformly distributed over a 2D circular sector.

    The radial coordinate is sampled so that the density is uniform in area
    within the annular sector defined by `r_range` and `theta_range`.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.
    r_range : tuple of float, optional
        Radial range `(r_min, r_max)`.
    theta_range : tuple of float, optional
        Angular range `(theta_min, theta_max)` in rad.

    Returns
    -------
    a1 : np.ndarray
        First Cartesian normalized coordinate, equal to
        `r_all * cos(theta_all)`.
    a2 : np.ndarray
        Second Cartesian normalized coordinate, equal to
        `r_all * sin(theta_all)`.
    r_all : np.ndarray
        Radial coordinate for each generated point.
    theta_all : np.ndarray
        Angular coordinate in rad for each generated point.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp

        np.random.seed(12345)

        a1, a2, r, theta = xp.generate_2D_uniform_circular_sector(
            num_particles=4,
            r_range=(1.0, 2.0),
            theta_range=(0.0, np.pi / 2))

        a1     # [1.222453, 0.828498, 0.0694, 0.65832]
        a2     # [1.514746, 1.123707, 1.24376, 1.086414]
        r      # [1.946496, 1.396111, 1.245695, 1.270307]
        theta  # [0.89178, 0.935479, 1.515056, 1.026008]
    """

    # CDF(r) = (r^2 - r0^2)/(r1^2 - r0^2)
    # InvCDF(u) = sqrt(r0^2 + u * (r1^2 -r0^2))

    r0 = r_range[0]
    r1 = r_range[1]

    uu = np.random.uniform(low=0, high=1., size=num_particles)

    r_all = np.sqrt(r0*r0 + uu * (r1*r1 - r0*r0))

    theta_all = np.random.uniform(low=theta_range[0], high=theta_range[1],
                           size=num_particles)

    a1 = r_all*np.cos(theta_all)
    a2 = r_all*np.sin(theta_all)

    return a1, a2, r_all, theta_all
