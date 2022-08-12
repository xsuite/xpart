# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xpart as xp
import xobjects as xo
import xtrack as xt

def _check_consistency_energy_variables(particles):

    # Check consistency between beta0 and gamma0
    assert np.allclose(particles.gamma0, 1/np.sqrt(1 - particles.beta0**2),
                       rtol=1e-14, atol=1e-14)

    # Assert consistency of p0c
    assert np.allclose(particles.p0c,
                       particles.mass0 * particles.beta0 * particles.gamma0,
                       rtol=1e-14, atol=1e-14)

    # Check energy0 property (consistency of p0c and gamma0)
    assert np.allclose(particles.energy0, particles.mass0 * particles.gamma0,
                       atol=1e-14, rtol=1e-14)

    # Check consistency of rpp and delta
    assert np.allclose(particles.rpp, 1./(particles.delta + 1),
                       rtol=1e-14, atol=1e-14)

    beta = particles.beta0 * particles.rvv
    gamma = 1/np.sqrt(1 - beta**2)
    pc = particles.mass0 * gamma * beta

    # Check consistency of delta with rvv
    assert np.allclose(particles.delta, (pc-particles.p0c)/(particles.p0c),
                       rtol=1e-14, atol=1e-14)

    # Check consistency of ptau with rvv
    energy = particles.mass0 * gamma
    assert np.allclose(particles.ptau, (energy - particles.energy0)/particles.p0c,
                       rtol=1e-14, atol=1e-14)

    # Check energy property
    assert np.allclose(particles.energy, energy, rtol=1e-14, atol=1e-14)


def test_basics():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context,
                mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
                x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
                zeta=[1e-2, 2e-2], delta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert dct['ptau'][0] == 0
        assert np.isclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

        particles = xp.Particles(_context=context,
                mass0=xp.PROTON_MASS_EV, q0=1, p0c=3e9,
                x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
                zeta=[1e-2, 2e-2], pzeta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert np.isclose(dct['ptau'][0], 0, atol=1e-14, rtol=0)
        assert np.isclose(dct['ptau'][1]/dct['beta0'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(dct['delta'][1], 9.99995545e-05, rtol=0, atol=1e-13)

        particles.move(_context=xo.ContextCpu())
        _check_consistency_energy_variables(particles)


def test_unallocated_particles():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context, _capacity=10,
        mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, # 7 TeV
        x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
        zeta=[1e-2, 2e-2], delta=[0, 1e-4])

        dct = particles.to_dict() # transfers it to cpu
        assert dct['x'][0] == 1e-3
        assert dct['ptau'][0] == 0
        assert np.isclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
        assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

        particles2 = xp.Particles.from_dict(dct, _context=context)


def test_linked_arrays():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        if isinstance(context, xo.ContextPyopencl):
            continue # Not supported

        ctx2np = context.nparray_from_context_array
        np2ctx = context.nparray_to_context_array
        particles = xp.Particles(_context=context, p0c=26e9, delta=[1,2,3])

        assert ctx2np(particles.delta[2]) == 3
        assert np.isclose(ctx2np(particles.rvv[2]), 1.00061, rtol=0, atol=1e-5)
        assert np.isclose(ctx2np(particles.rpp[2]), 0.25, rtol=0, atol=1e-10)
        assert np.isclose(ctx2np(particles.ptau[2]), 2.9995115176, rtol=0, atol=1e-6)

        particles.delta[1] = particles.delta[2]

        assert particles.delta[2] == particles.delta[1]
        assert particles.ptau[2] == particles.ptau[1]
        assert particles.rpp[2] == particles.rpp[1]
        assert particles.rvv[2] == particles.rvv[1]

        particles.ptau[0] = particles.ptau[2]

        assert particles.delta[2] == particles.delta[0]
        assert particles.ptau[2] == particles.ptau[0]
        assert particles.rpp[2] == particles.rpp[0]
        assert particles.rvv[2] == particles.rvv[0]

        particles = xp.Particles(_context=context, p0c=26e9,
                                 delta=[1,2,3,4,100,0])
        p0 = particles.copy()
        particles.state = np2ctx(np.array([1,1,1,1,0,1]))
        particles.delta[3:] = np2ctx([np.nan, 2, 3])

        assert particles.delta[5] == particles.delta[2]
        assert particles.ptau[5] == particles.ptau[2]
        assert particles.rvv[5] == particles.rvv[2]
        assert particles.rpp[5] == particles.rpp[2]

        assert particles.delta[4] == p0.delta[4]
        assert particles.ptau[4] == p0.ptau[4]
        assert particles.rvv[4] == p0.rvv[4]
        assert particles.rpp[4] == p0.rpp[4]

        assert particles.delta[3] == p0.delta[3]
        assert particles.ptau[3] == p0.ptau[3]
        assert particles.rvv[3] == p0.rvv[3]
        assert particles.rpp[3] == p0.rpp[3]

def test_sort():

    # Sorting available only on CPU for now

    p = xp.Particles(x=[0, 1, 2, 3, 4, 5, 6], _capacity=10)
    p.state[[0,3,4]] = 0

    tracker = xt.Tracker(line=xt.Line(elements=[xt.Cavity()]))
    tracker.track(p)

    assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.x == np.array([6, 1, 2, 5, 4, 3, 0,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort()

    assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(interleave_lost_particles=True)

    assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([0, 1, 1, 0, 0, 1, 1,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == -2
    assert p._num_lost_particles == -2

    p = xp.Particles(x=[6, 5, 4, 3, 2, 1, 0], _capacity=10)
    p.state[[0,3,4]] = 0

    tracker.track(p)

    assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.x == np.array([0, 5, 4, 1, 2, 3, 6,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(by='x')

    assert np.all(p.particle_id == np.array([6, 5, 2, 1, 4, 3, 0,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.x == np.array([0, 1, 4, 5, 2, 3, 6,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(by='x', interleave_lost_particles=True)

    assert np.all(p.particle_id == np.array([6, 5, 4, 3, 2, 1, 0,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.x == np.array([0, 1, 2, 3, 4, 5, 6,
                                    -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 0, 0, 1, 1, 0,
                                    -999999999, -999999999, -999999999]))
    assert p._num_active_particles == -2
    assert p._num_lost_particles == -2

def test_python_add_to_energy():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        particles = xp.Particles(_context=context,
        mass0=xp.PROTON_MASS_EV, q0=1, p0c=1.4e9,
        x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
        zeta=[1e-2, 2e-2], delta=[0, 1e-4])

        energy_before = particles.copy(_context=xo.ContextCpu()).energy
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta

        particles.add_to_energy(3e6)

        expected_energy = energy_before + 3e6
        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.energy, expected_energy,
                           atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.zeta == zeta_before)

def test_python_delta_setter():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0

        particles.delta = -2e-3

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.delta, -2e-3, atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.all(particles.px == px_before)
        assert np.all(particles.py == py_before)

def test_LocalParticle_add_to_energy():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        class TestElement(xt.BeamElement):
            _xofields={
                'value': xo.Float64,
                'pz_only': xo.Int64,
                }
            _extra_c_sources = ['''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    double const value = TestElementData_get_value(el);
                    int const pz_only = (int) TestElementData_get_pz_only(el);
                    //start_per_particle_block (part0->part)
                        LocalParticle_add_to_energy(part, value, pz_only);
                    //end_per_particle_block
                }
                ''']

        # pz_only = 1
        telem = TestElement(_context=ctx, value=1e6, pz_only=1)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        energy_before = particles.copy(_context=xo.ContextCpu()).energy
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.energy, energy_before + 1e6,
                           atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.all(particles.px == px_before)
        assert np.all(particles.py == py_before)

        # pz_only = 0
        telem = TestElement(_context=ctx, value=1e6, pz_only=0)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                 px = [1e-6, -1e-6], py = [2e-6, 0], zeta=0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        energy_before = particles.copy(_context=xo.ContextCpu()).energy
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        rpp_before = particles.copy(_context=xo.ContextCpu()).rpp
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.energy, energy_before + 1e6,
                           atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        rpp_after = particles.copy(_context=xo.ContextCpu()).rpp
        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.allclose(particles.px, px_before*rpp_before/rpp_after,
                           atol=1e-14, rtol=1e-14)
        assert np.allclose(particles.py, py_before*rpp_before/rpp_after,
                           atol=1e-14, rtol=1e-14)

def test_LocalParticle_update_delta():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        class TestElement(xt.BeamElement):
            _xofields={
                'value': xo.Float64,
                }

            _extra_c_sources =['''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    double const value = TestElementData_get_value(el);
                    //start_per_particle_block (part0->part)
                        LocalParticle_update_delta(part, value);
                    //end_per_particle_block
                }
                ''']

        telem = TestElement(_context=ctx, value=-2e-3)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.delta, -2e-3, atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.all(particles.px == px_before)
        assert np.all(particles.py == py_before)

def test_LocalParticle_update_ptau():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        class TestElement(xt.BeamElement):
            _xofields={
                'value': xo.Float64,
                }

            _extra_c_sources = ['''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    double const value = TestElementData_get_value(el);
                    //start_per_particle_block (part0->part)
                        LocalParticle_update_ptau(part, value);
                    //end_per_particle_block
                }
                ''']

        telem = TestElement(_context=ctx, value=-2e-3)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.ptau, -2e-3, atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.all(particles.px == px_before)
        assert np.all(particles.py == py_before)


def test_LocalParticle_update_pzeta():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        class TestElement(xt.BeamElement):
            _xofields={
                'value': xo.Float64,
                }
            _extra_c_sources = ['''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    double const value = TestElementData_get_value(el);
                    //start_per_particle_block (part0->part)
                        double const pzeta = LocalParticle_get_pzeta(part);
                        LocalParticle_update_pzeta(part, pzeta+value);
                    //end_per_particle_block
                }
                ''']

        telem = TestElement(_context=ctx, value=-2e-3)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        ptau_before  = particles.copy(_context=xo.ContextCpu()).ptau
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose((particles.ptau - ptau_before)/particles.beta0,
                           -2e-3, atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.gamma0 == gamma0_before)
        assert np.all(particles.zeta == zeta_before)
        assert np.all(particles.px == px_before)
        assert np.all(particles.py == py_before)

def test_LocalParticle_update_p0c():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        class TestElement(xt.BeamElement):
            _xofields={
                'value': xo.Float64,
                }
            _extra_c_sources = ['''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    double const value = TestElementData_get_value(el);
                    //start_per_particle_block (part0->part)
                        LocalParticle_update_p0c(part, value);
                    //end_per_particle_block
                }
                ''']

        telem = TestElement(_context=ctx, value=1.5e9)

        particles = xp.Particles(_context=ctx, p0c=1.4e9, delta=[0, 1e-3],
                                px = [1e-6, -1e-6], py = [2e-6, 0], zeta = 0.1)
        _check_consistency_energy_variables(
                                    particles.copy(_context=xo.ContextCpu()))
        px_before = particles.copy(_context=xo.ContextCpu()).px
        py_before = particles.copy(_context=xo.ContextCpu()).py
        energy_before  = particles.copy(_context=xo.ContextCpu()).energy
        beta0_before = particles.copy(_context=xo.ContextCpu()).beta0
        p0c_before = particles.copy(_context=xo.ContextCpu()).p0c
        zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
        telem.track(particles)

        particles.move(_context=xo.ContextCpu())
        assert np.allclose(particles.p0c, 1.5e9, atol=1e-14, rtol=1e-14)
        assert np.allclose(particles.energy, energy_before, atol=1e-14, rtol=1e-14)

        _check_consistency_energy_variables(particles)

        assert np.all(particles.zeta == zeta_before*particles.beta0/beta0_before)
        assert np.all(particles.px == px_before*p0c_before/particles.p0c)
        assert np.all(particles.py == py_before*p0c_before/particles.p0c)