# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
from xpart.random_number_generator import RandomGenerator

def ruth_PDF(t, A, B):
    return (A/(t**2))*(np.exp(-B*t))

t0 = 0.001
t1 = 0.02
rA = 0.0012306225579197868
rB = 53.50625

def test_random_generation():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        part = xp.Particles(_context=ctx, p0c=6.5e12, x=[1,2,3])
        part._init_random_number_generator()

        class TestElement(xt.BeamElement):
            _xofields={
                'dummy': xo.Float64,
                'rng':   RandomGenerator
                }

            _depends_on = [RandomGenerator]

            _extra_c_sources = [
                '''
                    /*gpufun*/
                    void TestElement_track_local_particle(
                            TestElementData el, LocalParticle* part0){
                        RandomGeneratorData rng = TestElementData_getp_rng(el);
                        //start_per_particle_block (part0->part)
                            double rr = RandomGenerator_get_double_ruth(rng, part);
                            LocalParticle_set_x(part, rr);
                        //end_per_particle_block
                    }
                ''']
            def __init__(self, **kwargs):
                if '_xobject' not in kwargs:
                    kwargs.setdefault('rng', RandomGenerator())
                super().__init__(**kwargs)

        telem = TestElement(_context=ctx)
        telem.rng.rutherford_A = rA
        telem.rng.rutherford_B = rB
        telem.rng.rutherford_lower_val = t0
        telem.rng.rutherford_upper_val = t1

        telem.track(part)

        # Use turn-by turin monitor to acquire some statistics

        tracker = xt.Tracker(_buffer=telem._buffer,
                line=xt.Line(elements=[telem]))

        tracker.track(part, num_turns=1e6, turn_by_turn_monitor=True)

        for i_part in range(part._capacity):
            x = tracker.record_last_track.x[i_part, :]
            assert np.all(x[i_part]>=t0)
            assert np.all(x[i_part]<=t1)
            hstgm, bin_edges = np.histogram(x[i_part],  bins=50, range=(t0, t1), density=True)
            bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
            ruth = [ruth_PDF(t, rA, rB) for t in bin_centers ]
            assert np.allclose(hstgm, ruth, rtol=1e-10, atol=1E-2)


def test_direct_sampling():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')
        n_seeds = 3
        ran = RandomGenerator(_capacity=3e6)
        ran = xp.random_number_generator.RandomGenerator(_capacity=3e6,
            rutherford_A=rA, rutherford_B=rB, rutherford_lower_val=t0, rutherford_upper_val=t1)
        samples, _ = ran.sample(distribution='rutherford', n_samples=1e6, n_seeds=n_seeds)

        for i_part in range(n_seeds):
            assert np.all(samples[i_part]>=t0)
            assert np.all(samples[i_part]<=t1)
            hstgm, bin_edges = np.histogram(samples[i_part],  bins=50, range=(t0, t1), density=True)
            bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
            ruth = [ruth_PDF(t, rA, rB) for t in bin_centers ]
            assert np.allclose(hstgm, ruth, rtol=1e-10, atol=1E-2)

