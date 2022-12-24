# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging
import warnings

import numpy as np

import xobjects as xo
import xtrack.linear_normal_form as lnf

import xpart as xp # To get the right Particles class depending on pyheatail interface state

logger = logging.getLogger(__name__)

def _check_lengths(**kwargs):
    length = None
    for nn, xx in kwargs.items():
        if hasattr(xx, "__iter__"):
            if hasattr(xx, 'shape') and len(xx.shape) == 0:
                continue
            if length is None:
                length = len(xx)
            else:
                if length != len(xx):
                    raise ValueError(f"invalid length len({nn})={len(xx)}")
    if 'num_particles' in kwargs.keys():
        num_particles = kwargs['num_particles']
        if num_particles is not None and length is None:
            length = num_particles
        if num_particles is not None and length != num_particles:
            raise ValueError(
              f"num_particles={num_particles} is inconsistent with array length")

    if length is None:
        length = 1
    return length

def build_particles(_context=None, _buffer=None, _offset=None, _capacity=None,
                      mode=None,
                      particle_ref=None,
                      num_particles=None,
                      x=None, px=None, y=None, py=None, zeta=None, delta=None,
                      x_norm=None, px_norm=None, y_norm=None, py_norm=None,
                      tracker=None,
                      at_element=None,
                      match_at_s=None,
                      particle_on_co=None,
                      R_matrix=None,
                      W_matrix=None,
                      method=None,
                      nemitt_x=None, nemitt_y=None,
                      scale_with_transverse_norm_emitt=None,
                      weight=None,
                      particles_class=None,
                      **kwargs, # They are passed to the twiss
                    ):

    """
    Function to create particle objects from arrays containing physical or
    normalized coordinates.

    Arguments:

        - mode: choose between:

            - `set`: reference quantities including mass0, q0, p0c, gamma0,
              etc. are taken from the provided reference particle. Particles
              coordinates are set according to the provided input x, px, y, py,
              zeta, delta (zero is assumed as default for these variables).
            - `shift`: reference quantities including mass0, q0, p0c, gamma0,
              etc. are taken from the provided reference particle. Particles
              coordinates are set from the reference particles and shifted
              according to the provided input x, px, y, py, zeta, delta (zero
              is assumed as default for these variables).
            - `normalized_transverse`: reference quantities including mass0,
              q0, p0c, gamma0, etc. are taken from the provided reference
              particle. The longitudinal coordinates are set according to the
              provided input `zeta`, `delta` (zero is assumed as default value
              for these variable`. The transverse coordinates are computed from
              normalized values `x_norm`, `px_norm`, `y_norm`, `py_norm` using
              the closed-orbit information and the linear transfer map obtained
              from the `tracker` or provided by the user.

            The default mode is `set`. `normalized_transverse` is used if any
            of x_norm, px_norm, y_norm, pynorm is provided.
        - particle_ref: particle object defining the reference quantities
          (mass0, 0, p0c, gamma0, etc.). Its coordinates (x, py, y, py, zeta,
          delta) are ignored unless `mode`='shift' is selected.
        - num_particles: Number of particles to be generated (used if provided
          coordinates are all scalar)
        - x: x coordinate of the particles (default is 0).
        - px: px coordinate of the particles (default is 0).
        - y: y coordinate of the particles (default is 0).
        - py: py coordinate of the particles (default is 0).
        - zeta: zeta coordinate of the particles (default is 0).
        - delta: delta coordinate of the particles (default is 0).
        - x_norm: transverse normalized coordinate x (in sigmas) used in
            combination with the one turn matrix R_matrix and with the
            transverse emittances provided in the argument
            `scale_with_transverse_norm_emitt` to generate x, px, y, py (x, px,
            y, py cannot be provided if x_norm, px_norm, y_norm, py_norm are
            provided).
        - x_norm: transverse normalized coordinate x (in sigmas).
        - px_norm: transverse normalized coordinate px (in sigmas).
        - y_norm: transverse normalized coordinate y (in sigmas).
        - py_norm: transverse normalized coordinate py (in sigmas).
        - tracker: tracker object used to find the closed orbit and the
          one-turn matrix.
        - particle_on_co: Particle on closed orbit
        - R_matrix: 6x6 matrix defining the linearized one-turn map to be used
          for the transformation of the normalized coordinates into physical
          space.
        - W_matrix: 6x6 matrix with the eigenvalues of the one-turn map
          (R_matrix). If provided, the R_matrix can be omitted.
        - nemitt_x: transverse normalized emittance in x.
        - nemitt_y: transverse normalized emittance in y.
        - weight: weights to be assigned to the particles.
        - at_element: location within the line at which particles are generated.
          It can be an index or an element name. It can be given  only if
          `at_tracker` is provided and `transverse_mode` is "normalized".
        - match_at_s: s coordinate of a location in the drifts downstream the
          specified `at_element` at which the particles are generated before
          being backdrifted to the location specified by `at_element`.
          No active element can be present in between.
        - _context: xobjects context in which the particle object is allocated.

    """

    assert mode in [None, 'set', 'shift', 'normalized_transverse']
    Particles = xp.Particles # To get the right Particles class depending on pyheatail interface state

    if particles_class is not None:
        raise NotImplementedError

    # Deprecation warning for scale_with_transverse_norm_emitt
    if scale_with_transverse_norm_emitt is not None:
        warnings.warn(
            "scale_with_transverse_norm_emitt is deprecated. Use "
            "nemitt_x and nemitt_y instead.",
            DeprecationWarning)

    if (particle_ref is not None and particle_on_co is not None):
        raise ValueError("`particle_ref` and `particle_on_co`"
                " cannot be provided at the same time")

    if particle_on_co is None and particle_ref is None:
        if tracker is not None:
            particle_ref = tracker.particle_ref

    if particle_ref is None:
        assert particle_on_co is not None, (
            "`particle_ref` or `particle_on_co` must be provided!")
        particle_ref = particle_on_co

    if not isinstance(particle_ref._buffer.context, xo.ContextCpu):
        particle_ref = particle_ref.copy(_context=xo.ContextCpu())

    # Move other input parameters to cpu if needed
    # Generated by:
    # for nn in 'x px y py zeta delta x_norm px_norm y_norm py_norm'.split():
    #     print(f'{nn} = ({nn}.get() if hasattr({nn}, "get") else {nn})')
    x = (x.get() if hasattr(x, "get") else x)
    px = (px.get() if hasattr(px, "get") else px)
    y = (y.get() if hasattr(y, "get") else y)
    py = (py.get() if hasattr(py, "get") else py)
    zeta = (zeta.get() if hasattr(zeta, "get") else zeta)
    delta = (delta.get() if hasattr(delta, "get") else delta)
    x_norm = (x_norm.get() if hasattr(x_norm, "get") else x_norm)
    px_norm = (px_norm.get() if hasattr(px_norm, "get") else px_norm)
    y_norm = (y_norm.get() if hasattr(y_norm, "get") else y_norm)
    py_norm = (py_norm.get() if hasattr(py_norm, "get") else py_norm)

    if tracker is not None and tracker.iscollective:
        logger.warning('Ignoring collective elements in particles generation.')
        tracker = tracker._supertracker

    if zeta is None:
        zeta = 0

    if delta is None:
        delta = 0

    if not np.isscalar(delta):
        delta = np.array(delta)

    if not np.isscalar(zeta):
        zeta = np.array(zeta)

    # Compute ptau from delta
    beta0 = particle_ref._xobject.beta0[0]
    delta_beta0 = delta * beta0
    ptau_beta0 = (delta_beta0 * delta_beta0
                        + 2. * delta_beta0 * beta0 + 1.)**0.5 - 1.
    pzeta = ptau_beta0 / beta0 / beta0

    if (x_norm is not None or px_norm is not None
        or y_norm is not None or py_norm is not None):

        assert (x is  None and px is  None
                and y is  None and py is  None)

        if mode is None:
            mode = 'normalized_transverse'
        else:
            assert mode == 'normalized_transverse'

    if mode is None:
        mode = 'set'

    if mode == 'normalized_transverse':
        if x_norm is None: x_norm = 0
        if px_norm is None: px_norm = 0
        if y_norm is None: y_norm = 0
        if py_norm is None: py_norm = 0
    else:
        if x is None: x = 0
        if px is None: px = 0
        if y is None: y = 0
        if py is None: py = 0

    assert particle_ref._capacity == 1
    ref_dict = {
        'q0': particle_ref.q0,
        'mass0': particle_ref.mass0,
        'p0c': particle_ref.p0c[0],
        'gamma0': particle_ref.gamma0[0],
        'beta0': particle_ref.beta0[0],
    }
    part_dict = ref_dict.copy()

    if at_element is not None or match_at_s is not None:
        # Only this case is covered if not starting at element 0
        assert tracker is not None
        assert mode == 'normalized_transverse'
        if isinstance(at_element, str):
            at_element = tracker.line.element_names.index(at_element)
        assert R_matrix is None # Not clear if it is at the element or at start machine
        if particle_on_co is not None:
            assert particle_on_co._xobject.at_element == 0

    if match_at_s is not None:
        import xtrack as xt
        assert at_element is not None, (
            'If `match_at_s` is provided, `at_element` needs to be provided and'
            'needs to correspond to the corresponding element in the sequence'
        )
        # Match at a position where there is no marker and backtrack to the previous marker
        expected_at_element = np.where(np.array(
            tracker.line.get_s_elements())<=match_at_s)[0][-1]
        assert at_element == expected_at_element or (
                at_element < expected_at_element and
                      all([isinstance(tracker.line.element_dict[nn], xt.Drift)
                           or tracker.line.element_dict[nn].__class__.__name__.startswith('Limit')
                for nn in tracker.line.element_names[at_element:expected_at_element]])), (
            "`match_at_s` can only be placed in the drifts downstream of the "
            "specified `at_element`. No active element can be present in between."
            )
        (tracker_rmat, _
            ) = xt.twiss._build_auxiliary_tracker_with_extra_markers(
                tracker=tracker, at_s=[match_at_s], marker_prefix='xpart_rmat_')
        at_element_tracker_rmat = tracker_rmat.line.element_names.index(
                                                                 'xpart_rmat_0')
    else:
        tracker_rmat = tracker
        at_element_tracker_rmat = at_element

    if mode == 'normalized_transverse':

        if W_matrix is None and tracker is not None:
            if method is not None:
                kwargs['method'] = method
            tw = tracker_rmat.twiss(particle_on_co=particle_on_co,
                                    particle_ref=particle_ref,
                                    R_matrix=R_matrix, **kwargs)
            tw_state = tw.get_twiss_init(at_element=
                (at_element_tracker_rmat if at_element_tracker_rmat is not None else 0))

            WW = tw_state.W_matrix
            particle_on_co = tw_state.particle_on_co
        elif W_matrix is None and R_matrix is not None:
            WW, _, _ = lnf.compute_linear_normal_form(
                                R_matrix,
                                **kwargs)
        else:
            WW = W_matrix

        WWinv = np.linalg.inv(WW)

        num_particles = _check_lengths(num_particles=num_particles,
            zeta=zeta, delta=delta, x_norm=x_norm, px_norm=px_norm,
            y_norm=y_norm, py_norm=py_norm)

        if scale_with_transverse_norm_emitt is not None:
            assert len(scale_with_transverse_norm_emitt) == 2
            assert nemitt_x is None and nemitt_y is None, (
                "If `scale_with_transverse_norm_emitt` is provided, "
                "`nemitt_x` and `nemitt_y` should not be provided.")

            nemitt_x = scale_with_transverse_norm_emitt[0]
            nemitt_y = scale_with_transverse_norm_emitt[1]

        if nemitt_x is None:
            gemitt_x = 1
        else:
            gemitt_x = (nemitt_x / particle_ref._xobject.beta0[0]
                        / particle_ref._xobject.gamma0[0])

        if nemitt_y is None:
            gemitt_y = 1
        else:
            gemitt_y = (nemitt_y / particle_ref._xobject.beta0[0]
                        / particle_ref._xobject.gamma0[0])


        x_norm_scaled = np.sqrt(gemitt_x) * np.array(x_norm)
        px_norm_scaled = np.sqrt(gemitt_x) * np.array(px_norm)
        y_norm_scaled = np.sqrt(gemitt_y) * np.array(y_norm)
        py_norm_scaled = np.sqrt(gemitt_y) * np.array(py_norm)

        # Transform long. coordinates to normalized space
        XX_long = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX_long[4, :] = zeta - particle_on_co._xobject.zeta[0]
        XX_long[5, :] = pzeta - particle_on_co._xobject.ptau[0] / beta0

        XX_norm_scaled = np.dot(WWinv, XX_long)

        XX_norm_scaled[0, :] = x_norm_scaled
        XX_norm_scaled[1, :] = px_norm_scaled
        XX_norm_scaled[2, :] = y_norm_scaled
        XX_norm_scaled[3, :] = py_norm_scaled

        # Transform to physical coordinates
        XX = np.dot(WW, XX_norm_scaled)

        XX[0, :] += particle_on_co._xobject.x[0]
        XX[1, :] += particle_on_co._xobject.px[0]
        XX[2, :] += particle_on_co._xobject.y[0]
        XX[3, :] += particle_on_co._xobject.py[0]
        XX[4, :] += particle_on_co._xobject.zeta[0]
        XX[5, :] += particle_on_co._xobject.ptau[0] / beta0

    elif mode == 'set':

        if R_matrix is not None:
            logger.warning('R_matrix provided but not used in this mode!')

        num_particles = _check_lengths(num_particles=num_particles,
            zeta=zeta, delta=delta, x=x, px=px,
            y=y, py=py)

        XX = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX[0, :] = x
        XX[1, :] = px
        XX[2, :] = y
        XX[3, :] = py
        XX[4, :] = zeta
        XX[5, :] = pzeta

    elif mode == "shift":

        if R_matrix is not None:
            logger.warning('R_matrix provided but not used in this mode!')

        num_particles = _check_lengths(num_particles=num_particles,
            zeta=zeta, delta=delta, x=x, px=px,
            y=y, py=py)

        XX = np.zeros(shape=(6, num_particles), dtype=np.float64)
        XX[0, :] = x + particle_ref.x
        XX[1, :] = px + particle_ref.px
        XX[2, :] = y + particle_ref.y
        XX[3, :] = py + particle_ref.py
        XX[4, :] = zeta + particle_ref.zeta
        XX[5, :] = pzeta + particle_ref.ptau / beta0
    else:
        raise ValueError('What?!')

    part_dict['x'] = XX[0, :]
    part_dict['px'] = XX[1, :]
    part_dict['y'] = XX[2, :]
    part_dict['py'] = XX[3, :]
    part_dict['zeta'] = XX[4, :]
    part_dict['ptau'] = XX[5, :] * beta0

    part_dict['weight'] = np.zeros(num_particles, dtype=np.int64)

    if _context is None and _buffer is None and tracker is not None:
        _context = tracker._buffer.context

    particles = Particles(_context=_context, _buffer=_buffer, _offset=_offset,
                          _capacity=_capacity,**part_dict)

    particles.particle_id[:num_particles] = particles._buffer.context.nparray_to_context_array(
                                   np.arange(0, num_particles, dtype=np.int64))
    if weight is not None:
        particles.weight[:num_particles] = weight

    if match_at_s is not None:
        # Backtrack to at_element
        length_aux_drift = -match_at_s + tracker.line.get_s_position(at_element)
        assert length_aux_drift <= 0
        auxdrift = xt.Drift(length=length_aux_drift,
                            _context=tracker._buffer.context)
        auxdrift.track(particles)

    if at_element is not None:
        if match_at_s is not None:
            particles.s[:num_particles] = particle_on_co._xobject.s[0] + length_aux_drift
        else:
            assert particle_on_co.at_element[0] == at_element
            particles.s[:num_particles] = particle_on_co._xobject.s[0]
        particles.at_element[:num_particles] = at_element

        particles.start_tracking_at_element = at_element

    return particles
