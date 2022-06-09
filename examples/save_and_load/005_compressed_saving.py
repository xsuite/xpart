# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp

n_particles = 1000
part = xp.Particles(_capacity=3000,
            px=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
            y=np.random.uniform(low=-1e-3, high=1e-3, size=n_particles),
            py=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
            zeta=np.random.uniform(low=-1e-2, high=1e-2, size=n_particles),
            delta=np.random.uniform(low=-1e-4, high=1e-4, size=n_particles),
            p0c=7e12)


df_part = part.to_pandas()
df_part_compact = part.to_pandas(compact=True)

assert len(df_part) == 3000
assert len(df_part_compact) == 1000

# Check that underscored vars (if any) are all rng states
for df in [df_part, df_part_compact]:
    ltest = [nn for nn in df.keys() if nn.startswith('_')]
    assert np.all([nn.startswith('_rng') for nn in ltest])

for nn in ['_rng_s1', '_rng_s2', '_rng_s3', '_rng_s4',
            'beta0', 'gamma0', 'ptau', 'rpp', 'rvv']:
    assert nn in df_part.keys()
    assert nn not in df_part_compact.keys()

df_part.to_hdf('part.hdf', key='df', mode='w')
df_part_compact.to_hdf('part_compact.hdf', key='df', mode='w')
df_part.to_hdf('part_compressed.hdf', key='df', mode='w',
               complevel=4, complib='zlib')
df_part_compact.to_hdf('part_compact_compressed.hdf', key='df', mode='w',
                       complevel=4, complib='zlib')

# Measured sizes
# -rw-r--r--  1 giadarol  staff   618K  2 Feb 17:05 part.hdf
# -rw-r--r--  1 giadarol  staff   158K  2 Feb 17:05 part_compact.hdf
# -rw-r--r--  1 giadarol  staff    98K  2 Feb 17:05 part_compressed.hdf
# -rw-r--r--  1 giadarol  staff    74K  2 Feb 17:05 part_compact_compressed.hdf

import pandas as pd
for fname in[
          'part', 'part_compact', 'part_compressed', 'part_compact_compressed']:
    if 'compact' in fname:
        pref = part.remove_unused_space()
    else:
        pref = part
    part_from_pdhdf = xp.Particles.from_pandas(pd.read_hdf(fname +'.hdf'))
    for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'gamma0']:
        assert np.all(getattr(pref, kk) == getattr(part_from_pdhdf, kk))
