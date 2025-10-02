#################################################
# This code randomly samples 4D                 #
# q-Gaussian distributions (1<q<5/3) using the  #
# sampling methods of Batygin                   #
# https://doi.org/10.1016/j.nima.2004.10.029    #
# and the 4D q-Gaussian formula derived in      #
# https://cds.cern.ch/record/2912366?ln=en      #
# These distributions are non-factorizable.     #
#################################################

import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d


def generate_radial_distribution(q, beta, F):
    """
    Compute the 4D radial distribution function for a round q-Gaussian.
    This can be numerically unstable if extreme values of q, beta as the
    sample space needs to be increased (eg for very high q or beta)

    Parameters:
        q (float): q-parameter (1 < q < 5/3).
        beta (float): Scale parameter, (beta > 1)
        F: sample space parameter, can be user defined

    Returns:
        tuple: (f_F, F) where f_F is the radial distribution, and F is the radial coordinate array.
    """
    assert q > 1, "q must be greater than 1"
    assert q < 5/3, "q must be less than 5/3"
    assert beta > 0, "beta must be greater than 0"

    term1 = -(beta**2) * (q - 3) * (q**2 - 1) / 4 / np.pi**2
    if abs(q - 1) < 1e-2:
        term2 = -1 / (1 - q)
    else:
        term2 = gamma(q / (q - 1)) / gamma(1 / (q - 1))

    term3 = (1 + beta * (q - 1) * F) ** (1 / (1 - q) - 3 / 2)
    return term1 * term2 * term3, F


def generate_pdf(f_F, F):
    """
    Compute the PDF g(F) from f(F) using the Abel transform in 4D.
    Cleans up the boundaries and gives correct normalisation factor.

    Parameters:
        f_F (np.ndarray): Distribution array.
        F (np.ndarray): Radial coordinate array.

    Returns:
        np.ndarray: Transformed PDF g(F).
    """
    f_F = f_F.copy()
    f_F[0] = f_F[-1] = 0  # Boundary cleanup
    return np.pi**2 * f_F * F


def generate_cdf(g_F, F):
    """
    Compute the cumulative distribution function (CDF) of g(F).

    Parameters:
        g_F (np.ndarray): PDF values.
        F (np.ndarray): Radial coordinates.

    Returns:
        np.ndarray: CDF of g(F).
    """
    delta_F = np.diff(F, prepend=0)
    cdf = np.cumsum(g_F * delta_F) # todo: rewrite with scipy.integrate.quad(g_F, -np.inf, np.inf)
    cdf = np.clip(cdf, 0, np.inf)
    return cdf


def sample_from_inv_cdf(Np, cdf_g, F):
    """
    Sample F values from the inverse CDF of g(F).

    Parameters:
        Np (int): Number of particles to sample.
        cdf_g (np.ndarray): CDF of g(F).
        F (np.ndarray): Original F grid.

    Returns:
        np.ndarray: Sampled F values (F_G).
    """
    cdf_g /= cdf_g[-1]  # normalize
    uniform_samples = np.random.uniform(0, 1, Np)
    interpolator = interp1d(
        cdf_g, F, kind="nearest", bounds_error=False, fill_value=(F[0], F[-1])
    )
    return interpolator(uniform_samples)


def generate_random_a(F_G):
    """
    Generate A_x and A_y coordinates based on F_G distribution.

    Parameters:
        F_G (np.ndarray): Sampled F values.

    Returns:
        tuple: (A_x, A_y) arrays.
    """
    A_X_SQ = np.random.uniform(0, F_G)
    A_x = np.sqrt(A_X_SQ)
    A_y = np.sqrt(F_G - A_X_SQ)
    return A_x, A_y


def generate_round_4D_q_gaussian_normalised(q, beta, n_part, sample_space=None):
    """
    Generate particles sampled from a 4D round q-Gaussian distribution.

    Parameters:
    -----------
    q : float
        The q parameter of the q-Gaussian distribution. Must satisfy 1 < q < 5/3.
    beta : float
        Scale parameter (analogous to beta function) for the distribution.
    n_part : int
        Number of particles to generate.
    sample_space : array_like, optional
        1D array of radius values at which to evaluate the radial PDF/CDF.
        If None, defaults to `np.linspace(0, 300, 1000000)`.

    Returns
    -------
    x : np.ndarray
        Horizontal position coordinates.
    px : np.ndarray
        Horizontal momentum coordinates.
    y : np.ndarray
        Vertical position coordinates.
    py : np.ndarray
        Vertical momentum coordinates.
    """
    if sample_space is None:
        F = np.linspace(0, 3e2, 1e6)
    else:
        F = sample_space

    f_F, F = generate_radial_distribution(q, beta, F)  # 4D distribution
    g_F = generate_pdf(f_F, F)  # PDF of 4D distribution
    cdf_g = generate_cdf(g_F, F)  # CDF
    F_G = sample_from_inv_cdf(n_part, cdf_g, F)  # Inverse function
    A_x, A_y = generate_random_a(F_G)  # random generator distributed like F_G

    # Sample angles for all particles
    beta_x = np.random.uniform(0, 2 * np.pi, n_part)
    beta_y = np.random.uniform(0, 2 * np.pi, n_part)

    # Compute positions and momenta
    x = A_x * np.cos(beta_x)
    px = -A_x * np.sin(beta_x)
    y = -A_y * np.cos(beta_y)
    py = A_y * np.sin(beta_y)

    return x, px, y, py


