# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_build_particles_normalized(test_context):
    for ctx_ref in [test_context, None]:
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=ctx_ref)


        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=test_context, line=xt.Line.from_dict(input_data['line']))

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       tracker=tracker, particle_ref=p0,
                                       zeta=0, delta=1e-3,
                                       x_norm=[1, 0, -1],  # in sigmas
                                       px_norm=[0, 1, 0],  # in sigmas
                                       nemitt_x=3e-6, nemitt_y=3e-6)

        dct = particles.to_dict() # transfers it to cpu
        assert np.allclose(dct['x'], [-0.0003883 , -0.0006076 , -0.00082689],
                           rtol=0, atol=1e-7)
        assert np.isclose(dct['ptau'][1], 1e-3, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-3, rtol=0, atol=1e-10)
        assert np.all(dct['p0c'] == 7e12)

        # Same as before with R matrix provided as input
        tw = tracker.twiss(particle_ref=p0)
        R_matrix = tw.R_matrix
        particle_on_co = tw.particle_on_co.copy()
        particles = xp.build_particles(_context=test_context,
                                       particle_on_co=particle_on_co,
                                       R_matrix=R_matrix,
                                       zeta=0, delta=1e-3,
                                       x_norm=[1, 0, -1],  # in sigmas
                                       px_norm=[0, 1, 0],  # in sigmas
                                       nemitt_x=3e-6, nemitt_y=3e-6)
        dct = particles.to_dict() # transfers it to cpu
        assert np.allclose(dct['x'], [-0.0003883, -0.0006076, -0.00082689],
                           rtol=0, atol=1e-7)
        assert np.isclose(dct['ptau'][1], 1e-3, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-3, rtol=0, atol=1e-10)
        assert np.all(dct['p0c'] == 7e12)

        # Test the 4d mode
        for ee in tracker.line.elements:
            if isinstance(ee, xt.Cavity):
                ee.voltage = 0

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context, method='4d',
                                       tracker=tracker, particle_ref=p0,
                                       zeta=0, delta=1e-3,
                                       x_norm=[1,0,-1], # in sigmas
                                       px_norm=[0,1,0], # in sigmas
                                       nemitt_x=3e-6, nemitt_y=3e-6)

        dct = particles.to_dict() # transfers it to cpu
        assert np.allclose(dct['x'], [-0.00038813 , -0.00060738 , -0.00082664],
                           rtol=0, atol=1e-7)
        assert np.isclose(dct['ptau'][1], 1e-3, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-3, rtol=0, atol=1e-10)
        assert np.all(dct['p0c'] == 7e12)


@for_all_test_contexts
def test_build_particles_normalized_closed_orbit(test_context):
    for ctx_ref in [test_context, None]:
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=ctx_ref)


        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=test_context, line=xt.Line.from_dict(input_data['line']))

        particle_co_guess = xp.build_particles(particle_ref=p0)
        particle_on_co = tracker.find_closed_orbit(particle_co_guess=particle_co_guess)

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       tracker=tracker, particle_ref=p0,
                                       zeta=particle_on_co._xobject.zeta[0],
                                       delta=particle_on_co._xobject.delta[0],
                                       x_norm=0, # in sigmas
                                       px_norm=0, # in sigmas
                                       nemitt_x=3e-6, nemitt_y=3e-6)

        dct = particles.to_dict()
        dct_co = particle_on_co.to_dict()

        for nn in 'x px y py zeta delta ptau rvv rpp gamma0 beta0 p0c'.split():
            assert np.allclose(dct[nn], dct_co[nn], atol=1e-15, rtol=0)


@for_all_test_contexts
def test_build_particles_normalized_match_at_s(test_context):
    for ctx_ref in [test_context, None]:
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=ctx_ref)

        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=test_context, line=xt.Line.from_dict(input_data['line']))

        at_element = 'ip6'
        i_start = tracker.line.element_names.index(at_element)
        s_start = tracker.line.get_s_position(i_start)
        # Get the last contigous Drift or Marker after IP6 with non-zero length
        i = i_start
        while True:
            i += 1
            if tracker.line[i].__class__.__name__ not in ['Drift', 'Marker']:
                break
        i_end = i
        ll = 0
        while ll==0:
            i -= 1
            ll = tracker.line[i].length
        match_at_s = tracker.line.get_s_position(i) + 0.76*ll

        # Ensure there is a Marker between at_element and match_at_s (to test behave_likes_drift)
        # TODO: this won't be needed once test lattices contain Markers
        tracker.line.insert_element(element=xt.Marker, name='test_marker', at_s=s_start + 0.7*(match_at_s-s_start))

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       tracker=tracker, particle_ref=p0,
                                       x=0.02, # in meters
                                       px_norm=np.random.normal(scale=0.1, size=300), # in sigmas
                                       y_norm=np.random.normal(scale=0.1, size=300),  # in sigmas
                                       py_norm=np.random.normal(scale=0.1, size=300), # in sigmas
                                       zeta=np.random.normal(scale=0.05, size=300),   # in meters
                                       delta=np.random.normal(scale=1e-4, size=300),
                                       nemitt_x=3e-6, nemitt_y=3e-6,
                                       at_element=at_element, match_at_s=match_at_s)

        assert not np.allclose(particles.x, 0.02, atol=1e-5)
        tracker.line.insert_element(element=xt.Marker, name='match_at_s', at_s=match_at_s)
        tracker.track(particles, num_elements=(i_start - i_end + 1))
#         print(particles.at_element)
#         print(tracker.line[particles.at_element])
        assert np.allclose(particles.x, 0.02, atol=1e-5)


