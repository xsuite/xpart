# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xpart as xp
import xtrack as xt

num_particles = 10000
nemitt_x = 2.5e-6
nemitt_y = 3e-6

# Load machine model
filename = ('../../../xtrack/test_data/hllhc15_noerrors_nobb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
tracker = xt.Tracker(line=xt.Line.from_dict(input_data['line']))
tracker.particle_ref = xp.Particles.from_dict(input_data['particle'])

# Location of the collimator
at_element = 'tcp.d6l7.b1'
at_s = tracker.line.get_s_position(at_element) + 0.1
x_cut = 3e-3 # position of the jaw
pencil_dr_sigmas = 0.7

# I generate a particle exactly on the jaw with no normalized px

tw_at_s = tracker.twiss(at_s=at_s)

p_on_cut = tracker.build_particles(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                   x=x_cut, px_norm=0, y_norm=0, py_norm=0,
                                   zeta_norm=0, pzeta_norm=0,
                                   at_element=at_element, match_at_s=at_s)

# Get accurate cut in sigmas
p_on_cut_norm = tw_at_s.get_normalized_coordinates(p_on_cut,
                                        nemitt_x=nemitt_x, nemitt_y=nemitt_y)



