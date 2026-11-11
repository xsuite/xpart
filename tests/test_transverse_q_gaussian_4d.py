import numpy as np
from scipy.optimize import curve_fit

from xpart.transverse_generators.q_gaussian_round import (
    generate_round_4D_q_gaussian_normalised)

def test_transverse_q_gaussian_4d_sample_std():
    """
    test standard deviation is correct
    """
    q = 1.4
    beta = 1
    n_part = 10000000
    x, _, _, _ = generate_round_4D_q_gaussian_normalised(q, beta, n_part)

    # variance a function of q and beta
    sample_variance = np.std(x)
    expected_variance = np.sqrt(1 / (beta * (5 - 3 * q)))

    assert np.isclose(sample_variance, expected_variance, atol=0.05), \
        f"Sample variance {sample_variance} deviates from expected {expected_variance}"

def q_gaussian_1d(x, q, beta, normalize=False):
    """
    Args:
        x:
        q: q-parameter
        beta: beta for q-Gaussian
        normalize: if normalize area to 1

    Returns:
        q-Gaussian function defined on x

    """
    assert q < 5/3, "q must be less than 5/3"
    arg = 1 - (1 - q) * beta * x**2
    f = np.where(arg > 0, arg**(1 / (1 - q)), 0)
    if normalize:
        dx = x[1] - x[0]
        area = np.sum(f) * dx
        f /= area
    return f

def test_transverse_q_gaussian_4d_sampler_returns_q0():
    """
    test that required q matches fitted q from scipy.optimize.curve fit
    """
    q = 1.4
    beta = 1
    n_part = 10000000
    x, _, _, _ = generate_round_4D_q_gaussian_normalised(q, beta, n_part)
    # Generate histogram
    bins = np.linspace(-10, 10, 1000)  # 1000 bins between -10 and 10
    counts, bin_edges = np.histogram(x, bins=bins, density=True)  # density=True to normalize to PDF

    # Calculate bin centers for fitting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    popt, pcov = curve_fit(q_gaussian_1d, bin_centers, counts, p0=[1.5, 1, 1.0])

    assert np.isclose(popt[0], q, atol=0.05), \
        f"Fitted q from samples {popt[0]} does not match requested q {q} "


# test that variable F(q, beta) works?


