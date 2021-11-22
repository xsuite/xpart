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
