# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from .polar import generate_2D_uniform_circular_sector
from ..general import _print

import xpart as xp

def generate_2D_pencil(num_particles, pos_cut_sigmas, dr_sigmas,
                       side='+'):

    '''
    Generate a 2D pencil beam distribution.

    Parameters
    ----------
    num_particles : int
        Number of particles to be generated.
    pos_cut_sigmas : float
        Position cut in sigmas.
    dr_sigmas : float
        Radius of the pencil beam in sigmas.
    side : str
        Side of the pencil beam. Can be '+', '-' or '+-'.

    Returns
    -------
    x1 : np.ndarray
        First normalized coordinate.
    x2 : np.ndarray
        Second normalized coordinate.

    '''

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
        assert line.tracker is not None

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
        drift_to_at_s = xt.Drift(_context=line._context,
            length=match_at_s - line.get_s_position(at_element))
    else:
        drift_to_at_s = None

    if twiss is None:
        twiss = line.twiss(at_s=match_at_s,
            at_elements=([at_element] if match_at_s is None else None),
            **kwargs)

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
        pencil_cut_sigmas = np.abs(p_on_cut_norm.x_norm)
    else:
        pencil_cut_sigmas = np.abs(p_on_cut_norm.y_norm)

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

