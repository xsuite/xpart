import numpy as np
from .linear_normal_form import compute_linear_normal_form

import xtrack as xt

def _check_lengths(**kwargs):
    length = None
    for nn, xx in kwargs.items():
        if hasattr(xx, "__iter__"):
            if length is None:
                length = len(xx)
            else:
                if length != len(xx):
                    raise ValueError(f"invalid length len({nn})={len(xx)}")
    return length

def assemble_particles(_context=None, _buffer=None, _offset=None,
                      particle_class=xt.Particles, particle_on_co=None,
                      x=None, px=None, y=None, py=None, zeta=None, delta=None,
                      x_norm=None, px_norm=None, y_norm=None, py_norm=None,
                      R_matrix=None,
                      weight=None):

    if not isinstance(particle_on_co, particle_class):
        particle_on_co = particle_class(**particle_on_co.to_dict())

    assert zeta is not None
    assert delta is not None

    if (x_norm is not None or px_norm is not None
        or y_norm is not None or py_norm is not None):

        assert (x is  None and px is  None
                and y is  None and py is  None)
        mode = 'normalized'
    elif (x is not  None or px is not  None
          or y is not  None or py is not  None):
        assert (x_norm is None and px_norm is None
                and y_norm is None and py_norm is None)
        mode = 'not normalized'
    else:
        raise ValueError('Invalid input')

    if mode == 'normalized':

        num_particles = _check_lengths(
            zeta=zeta, delta=delta, x_norm=x_norm, px_norm=px_norm,
            y_norm=y_norm, py_norm=py_norm)

        WW, WWinv, Rot = compute_linear_normal_form(R_matrix)

        # Transform long. coordinates to normalized space
        XX_long = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX_long[4, :] = zeta
        XX_long[5, :] = delta

        XX_norm = np.dot(WWinv, XX_long)

        XX_norm[0, :] = x_norm
        XX_norm[1, :] = px_norm
        XX_norm[2, :] = y_norm
        XX_norm[3, :] = py_norm

        # Transform to physical coordinates
        XX = np.dot(WW, XX_norm)

    elif mode == 'not normalized':

        num_particles = _check_lengths(
            zeta=zeta, delta=delta, x=x, px=px,
            y=y, py=py)

        XX = np.zeros(shape=(6, num_particle), dtype=np.float64)
        XX[0, :] = x
        XX[1, :] = px
        XX[2, :] = y
        XX[3, :] = py
        XX[4, :] = zeta
        XX[5, :] = delta

    assert particle_on_co._capacity == 1
    part_on_co_dict = {nn: np.atleast_1d(vv)[0] for nn, vv in particle_on_co.to_dict().items()
                       if not nn.startswith('_')}
    part_on_co_dict['x'] += XX[0, :]
    part_on_co_dict['px'] += XX[1, :]
    part_on_co_dict['y'] += XX[2, :]
    part_on_co_dict['py'] += XX[3, :]
    part_on_co_dict['zeta'] += XX[4, :]
    part_on_co_dict['delta'] += XX[5, :]

    del(part_on_co_dict['psigma'])

    part_on_co_dict['weight'] = np.zeros(num_particles, dtype=np.int64)

    particles = xt.Particles(_context=_context, _buffer=_buffer, _offset=_offset,
                             **part_on_co_dict)
    particles.particle_id = np.arange(0, num_particles, dtype=np.int64)
    if weight is not None:
        particles.weight[:] = weight

    return particles
