# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp

p1 = xp.Particles(x=[1,2,3])
p2 = xp.Particles(x=[4, 5])
p3 = xp.Particles(x=6)

particles = xp.Particles.merge([p1,p2,p3])

print(particles.x) # gives [1. 2. 3. 4. 5. 6.] 