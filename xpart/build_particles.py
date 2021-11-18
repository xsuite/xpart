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
                      scale_with_transverse_norm_emitt=None,
                      weight=None):

    """
    Function to create particle objects from arrays containing physical or normalized coordinates.

    Arguments:

        - particle_ref: Reference particle to which the provided arrays with coordinates are added
        - x: Values to be added to particle_ref.x
        - px: Values to be added to particle_ref.px
        - y: Values to be added to particle_ref.y
        - py: Values to be added to particle_ref.py
        - zeta: Values to be added to particle_ref.zeta
        - delta: Values to be added to particle_ref.delta
        - x_norm: transverse normalized coordinate x (in sigmas) used in combination
            with the one turn matrix R_matrix and with the transverse emittances provided in the argument
            scale_with_transverse_norm_emitt to generate x, px, y, py (x, px, y, py cannot be provided if
            x_norm, px_norm, y_norm, py_norm are provided).
        - px_norm: transverse normalized coordinate px (in sigmas) used in combination
            with the one turn matrix R_matrix and with the transverse emittances provided in the argument
            scale_with_transverse_norm_emitt to generate x, px, y, py (x, px, y, py cannot be provided if
            x_norm, px_norm, y_norm, py_norm are provided).
        - y_norm: transverse normalized coordinate y (in sigmas) used in combination
            with the one turn matrix R_matrix and with the transverse emittances provided in the argument
            scale_with_transverse_norm_emitt to generate x, px, y, py (x, px, y, py cannot be provided if
            x_norm, px_norm, y_norm, py_norm are provided).
        - py_norm: transverse normalized coordinate py (in sigmas) used in combination
            with the one turn matrix R_matrix and with the transverse emittances provided in the argument
            scale_with_transverse_norm_emitt to generate x, px, y, py (x, px, y, py cannot be provided if
            x_norm, px_norm, y_norm, py_norm are provided).
        - R_matrix: 6x6 matrix defining the linearized one-turn map to be used for the transformation of
            the normalized coordinates into physical space.
        - scale_with_transverse_norm_emitt: Tuple of two elements defining the transverse normalized
            emittances used to rescale the provided transverse normalized coordinates (x, px, y, py).
        - weight: weights to be assigned to the particles.
        - _context: xobjects context in which the particle object is allocated.

    """

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
        assert scale_with_transverse_norm_emitt is None, (
                'Available only for normalized coordinates')

        if x is None: x = 0
        if px is None: px = 0
        if y is None: y = 0
        if py is None: py = 0


    if mode == 'normalized':

        num_particles = _check_lengths(
            zeta=zeta, delta=delta, x_norm=x_norm, px_norm=px_norm,
            y_norm=y_norm, py_norm=py_norm)

        if scale_with_transverse_norm_emitt is not None:
            assert len(scale_with_transverse_norm_emitt) == 2

            nemitt_x = scale_with_transverse_norm_emitt[0]
            nemitt_y = scale_with_transverse_norm_emitt[1]

            gemitt_x = nemitt_x/particle_ref.beta0/particle_ref.gamma0
            gemitt_y = nemitt_y/particle_ref.beta0/particle_ref.gamma0

            x_norm_scaled = np.sqrt(gemitt_x) * x_norm
            px_norm_scaled = np.sqrt(gemitt_x) * px_norm
            y_norm_scaled = np.sqrt(gemitt_y) * y_norm
            py_norm_scaled = np.sqrt(gemitt_y) * py_norm
        else:
            x_norm_scaled = x_norm
            px_norm_scaled = px_norm
            y_norm_scaled = y_norm
            py_norm_scaled = py_norm




        WW, WWinv, Rot = compute_linear_normal_form(R_matrix)

        # Transform long. coordinates to normalized space
        XX_long = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX_long[4, :] = zeta
        XX_long[5, :] = delta

        XX_norm_scaled = np.dot(WWinv, XX_long)

        XX_norm_scaled[0, :] = x_norm_scaled
        XX_norm_scaled[1, :] = px_norm_scaled
        XX_norm_scaled[2, :] = y_norm_scaled
        XX_norm_scaled[3, :] = py_norm_scaled

        # Transform to physical coordinates
        XX = np.dot(WW, XX_norm_scaled)

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
