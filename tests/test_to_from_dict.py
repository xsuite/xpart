import numpy as np

import xobjects as xo
import xpart as xp

def test_to_from_dict():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        # Create a Particles on your selected context (default is CPU)

        part = xp.Particles(_context=context, x=[1,2,3])

        # Save particles to dict
        dct = part.to_dict()

        # Load particles from dict
        part_from_dict = xp.Particles.from_dict(dct, _context=context)

        #!end-doc-part
        dct = part.to_dict()
        for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
            assert isinstance(dct[nn], np.ndarray)
            assert isinstance(getattr(part_from_dict, nn), context.nplike_array_type)

def test_to_dict_pyheadtail_interface():

    xp.enable_pyheadtail_interface()
    assert xp.Particles.__name__ == 'PyHtXtParticles'
    xp.disable_pyheadtail_interface()
    assert xp.Particles.__name__ == 'Particles'

    p = xp.pyheadtail_interface.pyhtxtparticles.PyHtXtParticles(x=[1,2,3])

    dct = p.to_dict()
    p1 = xp.Particles.from_dict(dct)

    assert p1.x[1] == 2

def test_to_pandas():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        n_particles = 1000
        part = xp.Particles(_capacity=3000, _context=context,
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

        import pandas as pd
        for df, pref in zip([df_part, df_part_compact],
                            [part, part.remove_unused_space()]):
            part_test = xp.Particles.from_pandas(df)
            for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'gamma0']:
                assert np.all(pref.to_dict()[kk] == part_test.to_dict()[kk])
