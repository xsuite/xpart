# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

def test_single_rf_harmonic_matcher_rms_and_profile():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")
    
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=ctx)
    
    
        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']))
    
        rms_bunch_length=0.10
        for distribution in ["gaussian", "parabolic"]:
            zeta, delta, matcher = xp.generate_longitudinal_coordinates(tracker=tracker, particle_ref=p0,
                                                             num_particles=1000000,
                                                             sigma_z=rms_bunch_length, distribution=distribution,
                                                             engine="single-rf-harmonic", return_matcher=True)
            
            tau_distr_y = matcher.tau_distr_y
            tau_distr_x = matcher.tau_distr_x
            dx = tau_distr_x[1] - tau_distr_x[0]
            hist, _  = np.histogram(zeta, range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.), bins=len(tau_distr_x))
            hist = hist / sum(hist) * sum(tau_distr_y)
    
            assert np.isclose(rms_bunch_length, np.std(zeta), rtol=1e-2, atol=1e-15)
            assert np.all(np.isclose(hist, tau_distr_y, atol=3.e-2, rtol=1.e-2))