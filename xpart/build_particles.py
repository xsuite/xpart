import numpy as np

import xobjects as xo

from .linear_normal_form import compute_linear_normal_form

from .particles import Particles

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

def build_particles(_context=None, _buffer=None, _offset=None,
                      particle_class=Particles, particle_ref=None,
                      x=None, px=None, y=None, py=None, zeta=None, delta=None,
                      x_norm=None, px_norm=None, y_norm=None, py_norm=None,
                      R_matrix=None,
                      weight=None):


    if not isinstance(particle_ref, particle_class):
        particle_ref = particle_class(**particle_ref.to_dict())

    # Working on CPU, particles transferred at the end 
    particle_ref = particle_ref.copy(_context=xo.context_default)

    if zeta is None:
        zeta = 0

    if delta is None:
        delta = 0

    if (x_norm is not None or px_norm is not None
        or y_norm is not None or py_norm is not None):

        assert (x is  None and px is  None
                and y is  None and py is  None)
        mode = 'normalized'

        if x_norm is None: x_norm = 0
        if px_norm is None: px_norm = 0
        if y_norm is None: y_norm = 0
        if py_norm is None: py_norm = 0
    else:
        mode = 'not normalized'

        if x is None: x = 0
        if px is None: px = 0
        if y is None: y = 0
        if py is None: py = 0


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

        XX = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX[0, :] = x
        XX[1, :] = px
        XX[2, :] = y
        XX[3, :] = py
        XX[4, :] = zeta
        XX[5, :] = delta

    assert particle_ref._capacity == 1
    part_on_co_dict = {nn: np.atleast_1d(vv)[0] for nn, vv
                       in particle_ref.to_dict().items()
                       if not nn.startswith('_')}
    part_on_co_dict['x'] += XX[0, :]
    part_on_co_dict['px'] += XX[1, :]
    part_on_co_dict['y'] += XX[2, :]
    part_on_co_dict['py'] += XX[3, :]
    part_on_co_dict['zeta'] += XX[4, :]
    part_on_co_dict['delta'] += XX[5, :]

    del(part_on_co_dict['psigma'])

    part_on_co_dict['weight'] = np.zeros(num_particles, dtype=np.int64)

    particles = Particles(_context=_context, _buffer=_buffer, _offset=_offset,
                             **part_on_co_dict)
    particles.particle_id = particles._buffer.context.nparray_to_context_array(
                                   np.arange(0, num_particles, dtype=np.int64))
    if weight is not None:
        particles.weight[:] = weight

    return particles
