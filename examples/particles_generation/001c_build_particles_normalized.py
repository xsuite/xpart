# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import xobjects as xo
import xpart as xp
import xtrack as xt

# Build a reference particle
p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3)

# Choose a context
ctx = xo.ContextCpu()

# Load machine model (from pymask)
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']))

# Built a set of three particles with different x coordinates
particles = xp.build_particles(_context=ctx,
                               tracker=tracker, particle_ref=p0,
                               zeta=0, delta=1e-3,
                               x_norm=[1,0,-1], # in sigmas
                               px_norm=[0,1,0], # in sigmas
                               scale_with_transverse_norm_emitt=(3e-6, 3e-6)
                               )
#!end-doc-part
