import numpy as np

import xobjects as xo
import xpart as xp

# Create a Particles on your selected context (default is CPU)
n_particles = 1000
part = xp.Particles(
            x=np.random.uniform(low=-1e-3, high=1e-3, size=n_particles),
            px=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
            y=np.random.uniform(low=-1e-3, high=1e-3, size=n_particles),
            py=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
            zeta=np.random.uniform(low=-1e-2, high=1e-2, size=n_particles),
            delta=np.random.uniform(low=-1e-4, high=1e-4, size=n_particles),
            p0c=7e12)


##############
# PANDAS/HDF #
##############

# Save particles to hdf file via pandas
import pandas as pd
df = part.to_pandas()
df.to_hdf('part.hdf', key='df', mode='w')

df_compact = part.to_pandas(compact=True)
df_compact.to_hdf('part_compact.hdf', key='df', mode='w')
df_compact.to_hdf('part_compact_compressed.hdf', key='df', mode='w', complevel=4, complib='zlib')

########################
# Check data integrity #
########################

part_from_pdhdf = xp.Particles.from_pandas(pd.read_hdf('part_compact.hdf'))

for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'psigma']:
    assert np.all(getattr(part, kk) == getattr(part_from_pdhdf, kk))