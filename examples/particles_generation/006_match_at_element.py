# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import xpart as xp
import xtrack as xt

# Load machine model and build tracker
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])
line.build_tracker()

# Match distribution at a given element
particles = tracker.build_particles(x_norm=[0,1,2], px_norm=[0,0,0], # in sigmas
                   nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                   at_element='ip2')

# Match distribution at a given s position (100m downstream of ip6)
particles = tracker.build_particles(x_norm=[0,1,2], px_norm=[0,0,0], # in sigmas
                   nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                   at_element='ip6',
                   match_at_s=tracker.line.get_s_position('ip6') + 100
                   )


