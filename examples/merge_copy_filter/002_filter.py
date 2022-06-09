# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp

p1 = xp.Particles(x=[1,2,3], px=[10, 20, 30])

mask = p1.x > 1

p2 = p1.filter(mask)

print(p2.x) # gives [2. 3.]
print(p2.px) # gives [20. 30.]

