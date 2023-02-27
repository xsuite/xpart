# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np
import pytest
import xtrack as xt

import xpart as xp
from xpart.test_helpers import flaky_assertions, retry
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
@pytest.mark.parametrize('scenario', ['ions', 'protons'])
@pytest.mark.parametrize('distribution', ['gaussian', 'parabolic'])
@retry(n_times=3)
def test_single_rf_harmonic_matcher_rms_and_profile(test_context, scenario, distribution):
    if scenario == "protons":
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=test_context)

        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=test_context, line=xt.Line.from_dict(input_data['line']))
        tracker.line.particle_ref = p0

    elif scenario == "ions":
        # Load machine model (spsion)
        filename = xt._pkg_root.parent.joinpath('test_data/sps_ions/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=test_context, line=xt.Line.from_dict(input_data))
    else:
        raise NotImplementedError

    rms_bunch_length = 0.10

    zeta, delta, matcher = xp.generate_longitudinal_coordinates(
        tracker=tracker,
        # particle_ref=p0,
        num_particles=1000000,
        sigma_z=rms_bunch_length,
        distribution=distribution,
        engine="single-rf-harmonic",
        return_matcher=True
    )
    tau = zeta / tracker.line.particle_ref._xobject.beta0[0]
    tau_distr_y = matcher.tau_distr_y
    tau_distr_x = matcher.tau_distr_x
    dx = tau_distr_x[1] - tau_distr_x[0]
    hist, _ = np.histogram(tau, range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.),
                           bins=len(tau_distr_x))
    hist = hist / sum(hist) * sum(tau_distr_y)

    with flaky_assertions():
        assert np.isclose(rms_bunch_length, np.std(zeta), rtol=2e-2, atol=1e-15)
        assert np.all(np.isclose(hist, tau_distr_y, atol=3.e-2, rtol=1.e-2))
