import json

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

def test_build_particles_normalized():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        for ctx_ref in [ctx, None]:
            # Build a reference particle
            p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                              delta=[10], _context=ctx_ref)


            # Load machine model (from pymask)
            filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
            with open(filename, 'r') as fid:
                input_data = json.load(fid)
            tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']))

            # Built a set of three particles with different x coordinates
            particles = xp.build_particles(_context=ctx,
                                           tracker=tracker, particle_ref=p0,
                                           zeta=0, delta=1e-3,
                                           x_norm=[1,0,-1], # in sigmas
                                           px_norm=[0,1,0], # in sigmas
                                           scale_with_transverse_norm_emitt=(3e-6, 3e-6)
                                           )

            dct = particles.to_dict() # transfers it to cpu
            assert np.allclose(dct['x'], [-0.0003883 , -0.0006076 , -0.00082689],
                               rtol=0, atol=1e-7)
            assert np.isclose(dct['psigma'][1], 1e-3, rtol=0, atol=1e-9)
            assert np.isclose(1/(dct['rpp'][1]) - 1, 1e-3, rtol=0, atol=1e-10)
            assert np.all(dct['p0c'] == 7e12)

def test_build_particles_normalized_closed_orbit():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        for ctx_ref in [ctx, None]:
            # Build a reference particle
            p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                              delta=[10], _context=ctx_ref)


            # Load machine model (from pymask)
            filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
            with open(filename, 'r') as fid:
                input_data = json.load(fid)
            tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']))

            particle_co_guess = xp.build_particles(particle_ref=p0)
            particle_on_co = tracker.find_closed_orbit(particle_co_guess=particle_co_guess)

            # Built a set of three particles with different x coordinates
            particles = xp.build_particles(_context=ctx,
                                           tracker=tracker, particle_ref=p0,
                                           zeta=particle_on_co._xobject.zeta[0],
                                           delta=particle_on_co._xobject.delta[0],
                                           x_norm=0, # in sigmas
                                           px_norm=0, # in sigmas
                                           scale_with_transverse_norm_emitt=(3e-6, 3e-6)
                                           )

            dct = particles.to_dict()
            dct_co = particle_on_co.to_dict()

            for nn in 'x px y py zeta delta psigma rvv rpp gamma0 beta0 p0c'.split():
                assert np.allclose(dct[nn], dct_co[nn], atol=1e-15, rtol=0)
