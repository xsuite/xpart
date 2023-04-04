# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
from itertools import product
import numpy as np

import xpart as xp
import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts
from xpart.test_helpers import flaky_assertions, retry


@for_all_test_contexts
@retry()
def test_pencil_with_absolute_cut(test_context):

    num_particles = 10000
    nemitt_x = 2.5e-6
    nemitt_y = 3e-6

    # Load machine model
    filename = xt._pkg_root.parent.joinpath('test_data/hllhc15_noerrors_nobb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
    line=xt.Line.from_dict(input_data['line'])
    line.build_tracker(_context=test_context)
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    # Location of the collimator
    at_element = 'tcp.6l3.b1' # High dispersion
    # at_element = 'tcp.d6l7.b1' # Low dispersion
    at_s = line.get_s_position(at_element) + 1.
    absolute_cut = 3e-3 # position of the jaw
    pencil_dr_sigmas = 3 # width of the pencil

    for plane, side in product(['x', 'y'], ['+', '-']):

        if side == '+':
            absolute_cut = 3e-3
        else:
            absolute_cut = -3e-3

        print(f'plane={plane} side={side}')

        # generate pencil beam in absolute coordinates
        v_absolute, pv_absolute = xp.generate_2D_pencil_with_absolute_cut(num_particles,
                            plane=plane, absolute_cut=absolute_cut, dr_sigmas=pencil_dr_sigmas,
                            side=side, line=line,
                            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            at_element=at_element, match_at_s=at_s)

        # Vertical plane: generate gaussian distribution in normalized coordinates
        w_in_sigmas, pw_in_sigmas = xp.generate_2D_gaussian(num_particles)

        # Longitudinal plane: generate gaussian distribution matched to bucket
        zeta, delta = xp.generate_longitudinal_coordinates(
                num_particles=num_particles, distribution='gaussian',
                sigma_z=10e-2, line=line)

        # Combine the three planes
        # (the normalized coordinates in y/py are blurred in order to preserve the geometric ones)
        if plane == 'x':
            particles = line.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x=v_absolute, px=pv_absolute,
                            y_norm=w_in_sigmas, py_norm=pw_in_sigmas,
                            zeta=zeta, delta=delta,
                            at_element=at_element, match_at_s=at_s)
        elif plane == 'y':
            particles = line.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm=w_in_sigmas, px_norm=pw_in_sigmas,
                            y=v_absolute, py=pv_absolute,
                            zeta=zeta, delta=delta,
                            at_element=at_element, match_at_s=at_s)

        # Drift to at_s position for checking
        drift_to_at_s = xt.Drift(_context=test_context,
                        length=at_s-line.get_s_position(at_element))
        drift_to_at_s.track(particles)

        particles.move(_context=xo.ContextCpu())
        # Checks

        tw_at_s = line.twiss(at_s=at_s)
        norm_coords = tw_at_s.get_normalized_coordinates(
                        particles, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                        _force_at_element=0)

        v = getattr(particles, plane)
        pv = getattr(particles, 'p'+plane)
        betv = getattr(tw_at_s, 'bet'+plane)[0]
        alfv = getattr(tw_at_s, 'alf'+plane)[0]
        nemitt_v = {'x': nemitt_x, 'y': nemitt_y}[plane]
        sigma_v = np.sqrt(betv*nemitt_v
            /particles._xobject.beta0[0]/particles._xobject.gamma0[0])

        with flaky_assertions():
            assert(np.isclose(np.min(np.abs(v)), abs(absolute_cut), atol=1e-7))
            assert(np.isclose(np.max(np.abs(v)), abs(absolute_cut) + sigma_v*pencil_dr_sigmas,
                rtol=1e-3, atol=0))

            i_tip = np.argmax(np.abs(v))
            assert np.isclose(pv[i_tip]/v[i_tip], -alfv/betv, atol=5e-4)

            if side == '+':
                assert np.all(v >= 0)
            else:
                assert np.all(v <= 0)

            other_plane = {'x': 'y', 'y': 'x'}[plane]
            w_norm = getattr(norm_coords, other_plane+'_norm')
            pw_norm = getattr(norm_coords, 'p'+other_plane+'_norm')

            assert np.allclose(w_in_sigmas, w_norm, 1e-12)
            assert np.allclose(pw_in_sigmas, pw_norm, 1e-12)

            assert(np.allclose(zeta, particles.zeta, atol=1e-12))
            assert(np.allclose(delta, particles.delta, atol=1e-12))

