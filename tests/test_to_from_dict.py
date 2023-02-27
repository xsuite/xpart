# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import pytest

import xpart as xp

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_to_from_dict(test_context):
    # Create a Particles on your selected context (default is CPU)
    part = xp.Particles(_context=test_context, x=[1, 2, 3])

    # Save particles to dict
    dct = part.to_dict()

    # Load particles from dict
    part_from_dict = xp.Particles.from_dict(dct, _context=test_context)

    #!end-doc-part
    dct = part.to_dict()
    for nn in 'x px y py zeta delta ptau rpp rvv gamma0 p0c'.split():
        assert isinstance(dct[nn], np.ndarray)
        assert isinstance(getattr(part_from_dict, nn), test_context.nplike_array_type)


def test_to_dict_pyheadtail_interface():
    xp.enable_pyheadtail_interface()
    assert xp.Particles.__name__ == 'PyHtXtParticles'
    xp.disable_pyheadtail_interface()
    assert xp.Particles.__name__ == 'Particles'

    p = xp.pyheadtail_interface.pyhtxtparticles.PyHtXtParticles(x=[1, 2, 3])

    dct = p.to_dict()
    p1 = xp.Particles.from_dict(dct)

    assert p1.x[1] == 2


@for_all_test_contexts
@pytest.mark.parametrize('compact', [False, True])
def test_to_pandas(test_context, compact):
    n_particles = 1000
    capacity = 3000
    part = xp.Particles(_capacity=capacity, _context=test_context,
                x=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
                px=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
                y=np.random.uniform(low=-1e-3, high=1e-3, size=n_particles),
                py=np.random.uniform(low=-1e-6, high=1e-6, size=n_particles),
                zeta=np.random.uniform(low=-1e-2, high=1e-2, size=n_particles),
                delta=np.random.uniform(low=-1e-4, high=1e-4, size=n_particles),
                p0c=7e12)

    df_part = part.to_pandas(compact=compact)

    if compact:
        assert len(df_part) == n_particles
    else:
        assert len(df_part) == capacity

    # Check that underscored vars (if any) are all rng states
    ltest = [nn for nn in df_part.keys() if nn.startswith('_')]
    assert np.all([nn.startswith('_rng') for nn in ltest])

    for nn in ['_rng_s1', '_rng_s2', '_rng_s3', '_rng_s4',
                'beta0', 'gamma0', 'ptau', 'rpp', 'rvv']:
        if compact:
            assert nn not in df_part.keys()
        else:
            assert nn in df_part.keys()

    if compact:
        part = part.remove_unused_space()

    part_test = xp.Particles.from_pandas(df_part)
    for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'gamma0']:
        assert np.all(part.to_dict()[kk] == part_test.to_dict()[kk])