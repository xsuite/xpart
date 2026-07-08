# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from .polar import generate_2D_uniform_circular_sector
from ..build_particles import _trasport_twiss_over_drift
from ..general import _print

import xpart as xp

def generate_2D_pencil(num_particles, pos_cut_sigmas, dr_sigmas,
                       side='+'):

    """
    Generate a 2D pencil distribution in normalized coordinates.

    The generated points lie outside a position cut and within a radial
    thickness `dr_sigmas`, expressed in units of the normalized beam size. For
    `side='+'`, the cut is applied on the positive side of the first
    coordinate; for `side='-'`, on the negative side. With `side='+-'`, the
    particles are split between the two sides.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.
    pos_cut_sigmas : float
        Position cut in units of sigma.
    dr_sigmas : float
        Radial thickness of the pencil distribution in units of sigma.
    side : {'+', '-', '+-'}, optional
        Side on which to generate the pencil distribution.

    Returns
    -------
    x_norm : np.ndarray
        First normalized coordinate.
    px_norm : np.ndarray
        Second normalized coordinate.
    r_points : np.ndarray
        Radial coordinate of the generated points.
    theta_points : np.ndarray
        Angular coordinate in rad of the generated points.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp

        np.random.seed(12345)

        x_norm, px_norm, r, theta = xp.generate_2D_pencil(
            num_particles=4,
            pos_cut_sigmas=6.0,
            dr_sigmas=0.1,
            side='-')

        x_norm  # [-6.042132, -6.023206, -6.015881, -6.002956]
        px_norm # [0.785794, 0.322182, 0.178111, 0.46048]
        r       # [6.093015, 6.031817, 6.018517, 6.020591]
        theta   # [3.012266, 3.088153, 3.111994, 3.065034]
    """

    assert side == '+' or side == '-' or side == '+-'

    if side == '+-':
        n_plus = int(num_particles/2)
        n_minus = num_particles - n_plus
        x_plus, px_plus, r_plus, theta_plus = generate_2D_pencil(n_plus,
                                                 pos_cut_sigmas, dr_sigmas, side='+')
        x_minus, px_minus, r_minus, theta_minus = generate_2D_pencil(n_minus,
                                                 pos_cut_sigmas, dr_sigmas, side='-')
        x_norm = np.concatenate([x_minus, x_plus])
        px_norm = np.concatenate([px_minus, px_plus])
        r_points = np.concatenate([r_minus, r_plus])
        theta_points = np.concatenate([theta_minus, theta_plus])

        return x_norm, px_norm, r_points, theta_points

    else:

        r_min = np.abs(pos_cut_sigmas)
        r_max = r_min + dr_sigmas
        theta_max = np.arccos(pos_cut_sigmas/(np.abs(pos_cut_sigmas) + dr_sigmas))
        target_area = r_max**2/2*(2 * theta_max - np.sin(2 * theta_max))
        generated_area = (r_max**2 - r_min**2) * theta_max

        n_gen = int(num_particles*generated_area/target_area)

        x_gen, px_gen, r_points, theta_points = generate_2D_uniform_circular_sector(
                # Should avoid regen most of the times
                num_particles=int(n_gen*1.5+100),
                r_range=(r_min, r_max),
                theta_range=(-theta_max, theta_max))

        mask = x_gen > r_min
        x_norm = x_gen[mask]
        px_norm = px_gen[mask]
        r_points = r_points[mask]
        theta_points = theta_points[mask]

        assert len(x_norm) > num_particles

        x_norm = x_norm[:num_particles]
        px_norm = px_norm[:num_particles]
        r_points = r_points[:num_particles]
        theta_points = theta_points[:num_particles]

        if side == '-':
            x_norm = -x_norm
            theta_points = np.arctan2(px_norm, x_norm)

        return x_norm, px_norm, r_points, theta_points


def generate_2D_pencil_with_absolute_cut(num_particles,
    plane, absolute_cut, dr_sigmas, side='+', tracker=None, line=None,
    nemitt_x=None, nemitt_y=None,
    at_element=None, match_at_s=None, twiss=None, **kwargs):

    '''
    Generate a 2D pencil beam distribution with an absolute cut.

    Parameters
    ----------
    line: xtrack.Line
        Line for which the coordinates are generated.
    num_particles : int
        Number of particles to be generated.
    plane : str
        Plane of the pencil beam. Can be 'x' or 'y'.
    absolute_cut : float
        Absolute cut in meters.
    dr_sigmas : float
        Radius of the pencil beam in sigmas.
    side : str
        Side of the pencil beam. Can be '+' or '-'.

    Returns
    -------
    x1 : np.ndarray
        First normalized coordinate.
    x2 : np.ndarray
        Second normalized coordinate.

    '''

    if line is not None and tracker is not None:
        raise ValueError(
            'line and tracker cannot be provided at the same time.')

    if tracker is not None:
        _print('Warning! '
            "The argument tracker is deprecated. Please use line instead.")
        line = tracker.line

    if line is not None:
        if not line._has_valid_tracker():
            line.build_tracker()

    # kwargs are passed to line.twiss

    assert side == '+' or side == '-'
    assert plane == 'x' or plane == 'y'
    assert line is not None

    if match_at_s is not None:
        assert at_element is not None

    if at_element is None:
        at_element = 0

    if match_at_s is not None:
        import xtrack as xt
        tt = line.get_table()
        drift_to_at_s = xt.Drift(_context=line._context,
            length=match_at_s - tt['s', at_element])
    else:
        drift_to_at_s = None

    if twiss is None:
        # twiss = line.twiss(at_elements=([at_element] if match_at_s is None else None), at_s=match_at_s, **kwargs)
        twiss = line.twiss(**kwargs).rows[at_element]
        if match_at_s is not None:
            tt = line.tracker._tracker_data_base._line_table
            ds = match_at_s - tt['s', at_element]
            assert ds > 0
            tw_init = twiss.get_twiss_init(at_element=at_element)
            twiss = _trasport_twiss_over_drift(line.env, tw_init, ds).rows['_end_point']
    if side=='+':
        assert twiss[plane][0] < absolute_cut, 'The cut is on the wrong side'
    else:
        assert twiss[plane][0] > absolute_cut, 'The cut is on the wrong side'

    # Generate a particle exactly on the jaw with no amplitude in other eigenvectors
    p_on_cut_at_element = line.build_particles(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        x={'x': absolute_cut, 'y': None}[plane],
        y={'x': None, 'y': absolute_cut}[plane],
        x_norm={'x': None, 'y': 0}[plane],
        y_norm={'x': 0, 'y': None}[plane],
        px_norm=0, py_norm=0,
        zeta_norm=0, pzeta_norm=0,
        at_element=at_element, match_at_s=match_at_s, **kwargs)
    if drift_to_at_s is not None:
        p_on_cut_at_s = p_on_cut_at_element.copy()
        drift_to_at_s.track(p_on_cut_at_s)
    else:
        p_on_cut_at_s = p_on_cut_at_element

    # Get cut in (accurate) sigmas
    p_on_cut_norm = twiss.get_normalized_coordinates(p_on_cut_at_s,
                                        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        _force_at_element=0 # the twiss has only this element
                                        )
    if plane == 'x':
        pencil_cut_sigmas = np.abs(p_on_cut_norm.x_norm)[0]
    else:
        pencil_cut_sigmas = np.abs(p_on_cut_norm.y_norm)[0]

    # Generate normalized pencil in the selected plane (here w is x or y according to plane)
    w_in_sigmas, pw_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=dr_sigmas,
                             side=side)

    # Generate geometric coordinates in the selected plane only
    # (by construction y_cut is preserved)
    p_pencil_at_element = line.build_particles(
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                    x_norm={'x': w_in_sigmas, 'y': None}[plane],
                    px_norm={'x': pw_in_sigmas, 'y': None}[plane],
                    y_norm={'x': None, 'y': w_in_sigmas}[plane],
                    py_norm={'x': None, 'y': pw_in_sigmas}[plane],
                    zeta_norm=0, pzeta_norm=0,
                    at_element=at_element, match_at_s=match_at_s, **kwargs)

    if drift_to_at_s is not None:
        p_pencil_at_s = p_pencil_at_element.copy()
        drift_to_at_s.track(p_pencil_at_s)
    else:
        p_pencil_at_s = p_pencil_at_element

    if plane=='x':
        return p_pencil_at_s.x, p_pencil_at_s.px
    else:
        return p_pencil_at_s.y, p_pencil_at_s.py
