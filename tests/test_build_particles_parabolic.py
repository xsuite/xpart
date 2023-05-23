# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt
import xobjects as xo

from xpart.longitudinal.generate_parabolic_longitudinal_distribution import parabolic_longitudinal_distribution

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_build_particles_parabolic(test_context):
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
        particles = parabolic_longitudinal_distribution(_context=test_context,
									num_particles=100,
									nemitt_x=3e-6, 
									nemitt_y=3e-6, 
									sigma_z=0.05,
									particle_ref=p0, 
									total_intensity_particles=1e10,
									line=line
									)

        dct = particles.to_dict() 
        assert np.all(dct['p0c'] == 7e12)
        tw = line.twiss(particle_ref=p0)




