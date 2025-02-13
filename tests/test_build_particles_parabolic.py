# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from xpart.longitudinal import generate_parabolic_longitudinal_coordinates

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_build_particles_parabolic(test_context):
    # Build a reference particle
    p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                      delta=[10], _context=test_context)

    # Parameters for the test
    num_part = 1000000

    # Load machine model (from pymask)
    filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.build_tracker(_context=test_context)

    # Built a set of three particles with different x coordinates
    zeta, delta, matcher = generate_parabolic_longitudinal_coordinates(num_particles=num_part,
                                                                       nemitt_x=3e-6,
                                                                       nemitt_y=3e-6,
                                                                       sigma_z=0.05,
                                                                       particle_ref=p0,
                                                                       line=line,
                                                                       return_matcher=True
                                                                       )

    # Test if longitudinal coordinates match with Single
    # Generate distribution from RF matcher
    p0.move(_context=xo.ContextCpu())
    tau = zeta / p0.beta0[0]
    tau_distr_y = matcher.tau_distr_y
    tau_distr_x = matcher.tau_distr_x
    dx = tau_distr_x[1] - tau_distr_x[0]
    hist, edges = np.histogram(tau,
                range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.),
                bins=len(tau_distr_x))
    hist = hist / sum(hist) * sum(tau_distr_y)

    assert np.all(np.isclose(hist, tau_distr_y, atol=5.e-2, rtol=1.e-2))
