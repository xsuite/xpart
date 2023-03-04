# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np
import pytest
import xtrack as xt
import xobjects as xo

import xpart as xp
from xpart.test_helpers import flaky_assertions, retry
from xobjects.test_helpers import for_all_test_contexts

@pytest.mark.parametrize('scenario', ['psb_injection', 'sps_ions', 'lhc_protons'])
@pytest.mark.parametrize('distribution', ['gaussian', 'parabolic'])
@retry(n_times=3)
def test_single_rf_harmonic_matcher_rms_and_profile_and_tune(
                                    scenario, distribution):
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        if scenario == "lhc_protons":
            # Build a reference particle
            p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                              delta=[10], _context=ctx)

            # Load machine model (from pymask)
            filename = xt._pkg_root.parent.joinpath(
                'test_data/lhc_no_bb/line_and_particle.json')
            with open(filename, 'r') as fid:
                input_data = json.load(fid)
            line = xt.Line.from_dict(input_data['line'])
            line.build_tracker(_context=ctx)
            line.particle_ref = p0
            rms_bunch_length = 0.10

        elif scenario == "sps_ions":
            # Load machine model (spsion)
            filename = xt._pkg_root.parent.joinpath(
                'test_data/sps_ions/line_and_particle.json')
            with open(filename, 'r') as fid:
                input_data = json.load(fid)
            line = xt.Line.from_dict(input_data)
            line.build_tracker(_context=ctx)
            rms_bunch_length = 0.10
        elif scenario == "psb_injection":
            # Load machine model (psb injection)
            filename = xt._pkg_root.parent.joinpath(
                'test_data/psb_injection/line_and_particle.json')
            with open(filename, 'r') as fid:
                input_data = json.load(fid)
            line = xt.Line.from_dict(input_data)
            line.build_tracker(_context=ctx)
            rms_bunch_length = 17.
        else:
            raise NotImplementedError

        zeta, delta, matcher = xp.generate_longitudinal_coordinates(
            line=line, # particle_ref=p0,
            num_particles=1000000,
            sigma_z=rms_bunch_length, distribution=distribution,
            engine="single-rf-harmonic", return_matcher=True)
        tau = zeta / line.particle_ref._xobject.beta0[0]
        tau_distr_y = matcher.tau_distr_y
        tau_distr_x = matcher.tau_distr_x
        dx = tau_distr_x[1] - tau_distr_x[0]
        hist, _ = np.histogram(tau,
                    range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.),
                    bins=len(tau_distr_x))
        hist = hist / sum(hist) * sum(tau_distr_y)

        twiss_tune = line.twiss()['qs']
        theoretical_synchrotron_tune = matcher.get_synchrotron_tune()
        print(twiss_tune, theoretical_synchrotron_tune)
        print(twiss_tune - theoretical_synchrotron_tune,
            (theoretical_synchrotron_tune - twiss_tune)
                /(theoretical_synchrotron_tune))

        assert np.isclose(theoretical_synchrotron_tune,
                          twiss_tune, rtol=3.e-3, atol=1.e-15)

        with flaky_assertions():
            assert np.isclose(rms_bunch_length, np.std(zeta),
                            rtol=2e-2, atol=1e-15)
            assert np.all(np.isclose(hist, tau_distr_y,
                            atol=3.e-2, rtol=1.e-2))
