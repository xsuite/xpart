# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt
import xobjects as xo

from xpart.longitudinal import parabolic_longitudinal_distribution

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_build_particles_parabolic(test_context):
    for ctx_ref in [test_context, None]:
        # Build a reference particle
        p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3,
                          delta=[10], _context=ctx_ref)

	# Parameters for the test 
        num_part = 1000000
        parabolic_parameter = 0.05

        # Load machine model (from pymask)
        filename = xt._pkg_root.parent.joinpath('test_data/lhc_no_bb/line_and_particle.json')
        with open(filename, 'r') as fid:
            input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])
        line.build_tracker(_context=test_context)
	
	# Built a set of three particles with different x coordinates
        particles, matcher = parabolic_longitudinal_distribution(
                                                        num_particles=num_part,
                                                        nemitt_x=3e-6, 
                                                        nemitt_y=3e-6, 
                                                        sigma_z=parabolic_parameter,
                                                        particle_ref=p0, 
                                                        total_intensity_particles=1e10,
                                                        line=line,
                                                        return_matcher=True
                                                                )

        dct = particles.to_dict() 
        assert np.all(dct['p0c'] == 7e12)
        tw = line.twiss(particle_ref=p0)

	# Test if longitudinal coordinates match with Single
	# Generate distribution from RF matcher
        tau = particles.zeta / p0.beta0[0]
        tau_distr_y = matcher.tau_distr_y
        tau_distr_x = matcher.tau_distr_x
        dx = tau_distr_x[1] - tau_distr_x[0]
        hist, edges = np.histogram(tau,
                    range=(tau_distr_x[0]-dx/2., tau_distr_x[-1]+dx/2.),
                    bins=len(tau_distr_x))
        hist = hist / sum(hist) * sum(tau_distr_y)

        assert np.all(np.isclose(hist, tau_distr_y, atol=5.e-2, rtol=1.e-2))

