# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import trapezoid

from xobjects.test_helpers import fix_random_seed
from xpart.longitudinal.single_rf_harmonic_matcher import SingleRFHarmonicMatcher


def make_matcher(rms_bunch_length, tau_lim=1., **kwargs):
    return SingleRFHarmonicMatcher(
        q0=1, voltage=1e6, length=1000, freq=.99 * c / (2 * tau_lim),
        p0c=7e12, slip_factor=3e-4, beta0=1,
        rms_bunch_length=rms_bunch_length, distribution='binomial', **kwargs)


@pytest.mark.parametrize('m', [1.5, 4.7, 10.])
@pytest.mark.parametrize('relative_rms', [.1, .5, .57])
@pytest.mark.parametrize('tau_lim', [.01, 100.])
def test_binomial_profile_rms(monkeypatch, m, relative_rms, tau_lim):
    # Isolate the profile fit from the stochastic phase-space transformation.
    monkeypatch.setattr(SingleRFHarmonicMatcher,
                        'transform_tau_distr_to_m_distr', lambda self: ([], []))
    sigma = relative_rms * tau_lim
    matcher = make_matcher(sigma, tau_lim=tau_lim, m=m,
                           n_points_in_distribution=10001)
    x, y = matcher.tau_distr_x, matcher.tau_distr_y
    assert np.all(np.isfinite(y))
    assert np.all(y >= 0)
    rms = np.sqrt(trapezoid(x**2 * y, x) / trapezoid(y, x))
    np.testing.assert_allclose(rms, sigma, rtol=1e-5)


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf, .58, 1.])
def test_binomial_invalid_bunch_length(monkeypatch, sigma):
    monkeypatch.setattr(SingleRFHarmonicMatcher,
                        'transform_tau_distr_to_m_distr', lambda self: ([], []))
    with pytest.raises(ValueError, match='rms_bunch_length'):
        make_matcher(sigma)


@pytest.mark.parametrize('m', [.5, 0., np.nan, np.inf])
def test_binomial_invalid_exponent(monkeypatch, m):
    monkeypatch.setattr(SingleRFHarmonicMatcher,
                        'transform_tau_distr_to_m_distr', lambda self: ([], []))
    with pytest.raises(ValueError, match='m must'):
        make_matcher(.1, m=m)


def fail_if_sampling_starts(*args, **kwargs):
    pytest.fail('Invalid distribution reached the rejection sampling loop')


def test_binomial_unresolved_profile(monkeypatch):
    matcher = make_matcher(1e-5)
    assert not np.any(matcher.tau_distr_y)
    monkeypatch.setattr(matcher, 'get_airbag_from_m', fail_if_sampling_starts)
    with pytest.raises(ValueError, match='distribution'):
        matcher.sample_tau_ptau(n_particles=1)


@pytest.mark.parametrize('x,y', [
    ([], []), ([0.], [0.]), ([0., .5], [0., 0.]),
    ([0., .5], [0., -1.]), ([0., .5], [0., np.nan]),
    ([0., .5], [0., np.inf]), ([0., np.nan], [0., 1.]),
])
def test_invalid_distribution_rejected_before_sampling(monkeypatch, x, y):
    matcher = SingleRFHarmonicMatcher.__new__(SingleRFHarmonicMatcher)
    matcher.m_distr_x, matcher.m_distr_y = x, y
    monkeypatch.setattr(matcher, 'get_airbag_from_m', fail_if_sampling_starts)
    with pytest.raises(ValueError, match='distribution'):
        matcher.sample_tau_ptau(n_particles=1)


@pytest.mark.parametrize('sigma', [.1, .5])
@fix_random_seed(12345)
def test_binomial_sampling(sigma):
    matcher = make_matcher(sigma, transformation_particles=100000)
    tau, ptau = matcher.sample_tau_ptau(n_particles=20000)
    assert len(tau) == len(ptau) == 20000
    assert np.all(np.isfinite(tau))
    assert np.all(np.isfinite(ptau))
    np.testing.assert_allclose(np.std(tau), sigma, rtol=.03)
