# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
import xobjects as xo

p1 = xp.Particles(x=[1,2,3])

# Make a copy of p1 in the same context
p2 = p1.copy()

# Alter p1
p1.x += 10

# Inspect
print(p1.x) # gives [11. 12. 13.]
print(p2.x) # gives [1. 2. 3.]

# Copy across contexts
ctxgpu = xo.ContextCupy()
p3 = p1.copy(_context=ctxgpu)

# Inspect
print(p3.x[2]) # gives 13
