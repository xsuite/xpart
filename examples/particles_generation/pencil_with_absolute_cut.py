import xtrack as xt
import xpart as xp

def generate_2D_pencil_with_absolute_cut(num_particles,
    plane, absolute_cut, dr_sigmas, side='+', tracker=None,
    nemitt_x=None, nemitt_y=None,
    at_element=None, match_at_s=None, **kwargs):

    assert side == '+' or side == '-'
    assert plane == 'x' or plane == 'y'
    assert tracker is not None

    if match_at_s is not None:
        assert at_element is not None

    if at_element is None:
        at_element = 0

    if match_at_s is not None:
        drift_to_at_s = xt.Drift(
            length=match_at_s - tracker.line.get_s_position(at_element))
    else:
        drift_to_at_s = None


    tw_at_s = tracker.twiss(at_s=match_at_s, at_elements=[at_element], **kwargs)

    # Generate a particle exactly on the jaw with no amplitude in other eigemvectors
    p_on_cut_at_element = tracker.build_particles(
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
    p_on_cut_norm = tw_at_s.get_normalized_coordinates(p_on_cut_at_s,
                                        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        _force_at_element=0 # the twiss has only this element
                                        )
    if plane == 'x':
        pencil_cut_sigmas = p_on_cut_norm.x_norm
    else:
        pencil_cut_sigmas = p_on_cut_norm.y_norm

    # Generate normalized pencil in the selected plane (here w is x or y according to plane)
    w_in_sigmas, pw_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=num_particles,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=dr_sigmas,
                             side=side)

    # Generate geometric coordinates in the selected plane only
    # (by construction y_cut is preserved)
    p_pencil_at_element = tracker.build_particles(
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


