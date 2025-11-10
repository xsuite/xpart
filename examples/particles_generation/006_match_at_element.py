# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xtrack as xt

line = xt.load('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
line.set_particle_ref('proton', p0c=7e12)

# Match distribution at a given element
particles = line.build_particles(x_norm=[0,1,2], px_norm=[0,0,0], # in sigmas
                   nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                   at_element='ip2')

# Match distribution at a given s position (100m downstream of ip6)
particles = line.build_particles(x_norm=[0,1,2], px_norm=[0,0,0], # in sigmas
                   nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                   at_element='ip6',
                   match_at_s=line.get_s_position('ip6') + 100
                   )


