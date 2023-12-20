# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt
import xobjects as xo

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
        line = xt.Line.from_dict(input_data['line'])
        line.build_tracker(_context=test_context)

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       line=line, particle_ref=p0,
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
        tw = line.twiss(particle_ref=p0)
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
        for ee in line.elements:
            if isinstance(ee, xt.Cavity):
                ee.voltage = 0

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context, method='4d',
                                       line=line, particle_ref=p0,
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
def test_build_particles_normalized_ions(test_context):
    for ctx_ref in [test_context, None]:
        filename = xt._pkg_root.parent / 'test_data' / 'sps_ions' / 'line_and_particle.json'
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        line = xt.Line.from_dict(input_data)
        p0 = line.particle_ref
        line.build_tracker(_context=test_context)

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       line=line, particle_ref=p0,
                                       zeta=0, delta=1e-5,
                                       x_norm=[1, 0, -1],  # in sigmas
                                       px_norm=[0, 1, 0],  # in sigmas
                                       nemitt_x=3e-6, nemitt_y=3e-6)

        dct = particles.to_dict() # transfers it to cpu
        assert np.allclose(dct['x'], [6.5404e-3, 1.21e-5, -6.5163e-3],
                           rtol=0, atol=1e-7)
        assert np.isclose(dct['ptau'][1], 9.906e-6, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-5, rtol=0, atol=1e-10)
        assert np.allclose(dct['p0c'], 1.4024063e+12, rtol=0, atol=1e3)


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
        line = xt.Line.from_dict(input_data['line'])
        line.build_tracker(_context=test_context)

        co_guess = xp.build_particles(particle_ref=p0)
        particle_on_co = line.find_closed_orbit(co_guess=co_guess)

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       line=line, particle_ref=p0,
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
        line = xt.Line.from_dict(input_data['line'])

        at_element = 'ip6'
        i_start = line.element_names.index(at_element)
        s_start = line.get_s_position(i_start)
        # Find first active element
        for iele in range(i_start, len(line)):
            ee = line[iele]
            if not isinstance(ee, (xt.Drift, xt.Marker)):
                i_next_active = iele
                break

        match_at_s = line.get_s_position(i_next_active) - 0.11

        # Ensure there is a Marker between at_element and match_at_s (to test behave_likes_drift)
        line.insert_element(element=xt.Marker(), name='test_marker', at_s=s_start + 0.3*(match_at_s-s_start))

        line.build_tracker(_context=test_context)

        # Built a set of three particles with different x coordinates
        particles = xp.build_particles(_context=test_context,
                                       line=line, particle_ref=p0,
                                       x=0.02, # in meters
                                       px_norm=np.random.normal(scale=0.1, size=300), # in sigmas
                                       y_norm=np.random.normal(scale=0.1, size=300),  # in sigmas
                                       py_norm=np.random.normal(scale=0.1, size=300), # in sigmas
                                       zeta=np.random.normal(scale=0.05, size=300),   # in meters
                                       delta=np.random.normal(scale=1e-4, size=300),
                                       nemitt_x=3e-6, nemitt_y=3e-6,
                                       at_element=at_element, match_at_s=match_at_s)

        particles.move(_context=xo.context_default)
        assert not np.allclose(particles.x, 0.02, atol=1e-20)
        line.unfreeze()
        line.insert_element(element=xt.Marker(), name='match_at_s', at_s=match_at_s)
        line.build_tracker(_context=test_context)
        particles.move(_context=test_context)
        line.track(particles, ele_stop='match_at_s')
        particles.move(_context=xo.context_default)
        assert (np.unique(particles.at_element[particles.state>0])[0]
                         == line.element_names.index('match_at_s'))
        assert np.allclose(particles.x, 0.02, atol=1e-20)


@for_all_test_contexts
def test_build_perticles_dispersion(test_context):

    # Load machine model (from pymask)
    filename = xt._pkg_root.parent.joinpath(
                                'test_data/lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)

    line=xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                               q0=1, p0c=7e12)
    line.build_tracker(_context=test_context)

    particles = line.build_particles(nemitt_x=3e-6, nemitt_y=3e-6,
                    x=[1e-3, -1e-3], x_norm=0, px_norm=0, y_norm=0, py_norm=0)


    tw = line.twiss()
    norm_coords = tw.get_normalized_coordinates(
                                    particles, nemitt_x=3e-6, nemitt_y=3e-6)
    particles.move(_context=xo.ContextCpu())

    assert np.allclose(particles.x, [1e-3, -1e-3], atol=1e-10, rtol=0)
    assert np.allclose((particles.x-tw.x[0])/particles.delta, tw.dx[0],
                       atol=5e-3, rtol=0)

    assert np.allclose(norm_coords['x_norm'], 0, 1e-12)
    assert np.allclose(norm_coords['px_norm'], 0, 1e-12)
    assert np.allclose(norm_coords['y_norm'], 0, 1e-12)
    assert np.allclose(norm_coords['py_norm'], 0, 1e-12)

    particles = line.build_particles(nemitt_x=3e-6, nemitt_y=3e-6,
                    x=[1e-3, -1e-3], x_norm=[0.3, 0.4], px_norm=[0.5, 0.6],
                    y_norm=[0.7, 0.8], py_norm=[0.9, 1.0])

    tw = line.twiss()
    norm_coords = tw.get_normalized_coordinates(
                                    particles, nemitt_x=3e-6, nemitt_y=3e-6)
    particles.move(_context=xo.ContextCpu())

    assert np.allclose(particles.x, [1e-3, -1e-3], atol=1e-10, rtol=0)

    assert np.allclose(norm_coords['x_norm'], [0.3, 0.4], 1e-12)
    assert np.allclose(norm_coords['px_norm'], [0.5, 0.6], 1e-12)
    assert np.allclose(norm_coords['y_norm'], [0.7, 0.8], 1e-12)
    assert np.allclose(norm_coords['py_norm'], [0.9, 1.0], 1e-12)
