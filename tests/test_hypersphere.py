# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy import stats

import xpart as xp


def test_hypersphere_2D():
    num_particles = int(20e4)
    r = 2.5

    means = []
    stds = []
    for seed in [1, 2, 3]:
        x_norm, px_norm = xp.generate_hypersphere_2D(
            num_particles, r=r, rng_seed=seed)

        data = np.array([x_norm, px_norm]).T
        bins = [np.linspace(-r, r, 50) for i in range(2)]

        bincounts = stats.binned_statistic_dd(
            data, 1, bins=bins, statistic='count').statistic
        bincounts = bincounts[bincounts != 0]

        means.append(np.mean(bincounts))
        stds.append(np.std(bincounts))

    assert np.allclose(np.diff(means)/np.mean(means), 0, rtol=0, atol=1e-2)
    assert np.allclose(np.array(stds)/np.array(means), 0, rtol=0, atol=0.5)


def test_hypersphere_4D():
    num_particles = int(20e4)
    rx = 2.5
    ry = 4.2

    means = []
    stds = []
    for seed in [1, 2, 3]:
        x_norm, px_norm, y_norm, py_norm = xp.generate_hypersphere_4D(
            num_particles, rx=rx, ry=ry, rng_seed=seed)

        data = np.array([x_norm, px_norm, y_norm, py_norm]).T
        bins = [np.linspace(-rx, rx, 50) for i in range(2)] + \
            [np.linspace(-ry, ry, 50) for i in range(2)]

        bincounts = stats.binned_statistic_dd(
            data, 1, bins=bins, statistic='count').statistic
        bincounts = bincounts[bincounts != 0]

        means.append(np.mean(bincounts))
        stds.append(np.std(bincounts))

    assert np.allclose(np.diff(means)/np.mean(means), 0, rtol=0, atol=1e-2)
    assert np.allclose(np.array(stds)/np.array(means), 0, rtol=0, atol=0.5)


def test_hypersphere_6D():
    num_particles = int(20e4)
    rx = 2.5
    ry = 4.2
    rzeta = 1.2

    means = []
    stds = []
    for seed in [1, 2, 3]:
        x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm = xp.generate_hypersphere_6D(
            num_particles, rx=rx, ry=ry, rzeta=rzeta, rng_seed=seed)

        data = np.array(
            [x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm]).T
        bins = ([np.linspace(-rx, rx, 20) for i in range(2)]
                + [np.linspace(-ry, ry, 20) for i in range(2)]
                + [np.linspace(-rzeta, rzeta, 20) for i in range(2)])

        bincounts = stats.binned_statistic_dd(
            data, 1, bins=bins, statistic='count').statistic
        bincounts = bincounts[bincounts != 0]

        means.append(np.mean(bincounts))
        stds.append(np.std(bincounts))

    assert np.allclose(np.diff(means)/np.mean(means), 0, rtol=0, atol=1e-2)
    assert np.allclose(np.array(stds)/np.array(means), 0, rtol=0, atol=0.5)
