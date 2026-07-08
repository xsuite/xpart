# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xtrack as xt
import xpart as xp


fname_line = '../../../xtrack/test_data/hllhc_14/line_and_particle.json'


##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.build_tracker()

particle_ref = xp.Particles.from_dict(input_data['particle'])

zeta, delta = line.xpart.generate_longitudinal_coordinates(
        num_particles=100000, distribution='gaussian',
        sigma_z=10e-2, particle_ref=particle_ref)

assert np.isclose(np.std(zeta), 10e-2, rtol=1e-2, atol=0)
assert np.isclose(np.std(delta), 1.21e-4, rtol=1e-2, atol=0)
