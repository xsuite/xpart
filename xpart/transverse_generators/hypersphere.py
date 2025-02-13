
import numpy as np




def generate_hypersphere(N, D, r=1, rng_seed = 0, surface=False ,unpack = False):
    '''
    Generate points uniformly distributed inside or on the surface of an N-dimensional hypersphere.
    Adapted from : https://baezortega.github.io/2018/10/14/hypersphere-sampling/

    Parameters
    ----------
    N : int
        Number of particles to generate.
    D : int
        Dimension of the hypersphere.
    r : float or list
        Radius of the hypersphere. If a list, specifies radii for anisotropic scaling.
    rng_seed : int
        Seed for the random number generator for reproducibility.
    surface : bool
        If True, points will be generated on the surface. If False, points will be generated inside the hypersphere.
    unpack : bool
        If True, returns individual arrays for each dimension. If False, returns a single array with shape (N, D).

    Returns
    -------
    samples : np.ndarray or tuple of np.ndarray
        Generated points. Shape is (N, D) if unpack is False, otherwise D arrays of shape (N,).
    '''
    # Set the random seed for reproducibility
    rng = np.random.default_rng(int(rng_seed))

    # Sample D vectors of N Gaussian coordinates
    N = int(N)
    D = int(D)
    samples = rng.standard_normal(size = (N, D))

    # Normalise all distances (radii) to 1
    radii = np.sqrt(np.sum(samples ** 2, axis=1))[:,np.newaxis]
    samples = samples / radii

    # Sample N radii with exponential distribution (unless points are to be on the surface)
    if not surface:
        new_radii = np.random.uniform(low=0.0, high=1.0, size=(N, 1)) ** (1 / D)
        samples = samples * new_radii

    # Scale the samples to the desired radius
    if isinstance(r,list):
        r = np.array(r)[np.newaxis,:]
    elif isinstance(r,type(np.array([]))):
        assert False, 'r should be float or list'
    samples = samples * r

    if not unpack:
        return samples
    else:
        return samples.T


def generate_hypersphere_2D(num_particles,r = 1, rng_seed = 0):
    '''
    Generate points uniformly distributed inside a 2-dimensional hypersphere (circle).

    Parameters
    ----------
    num_particles : int
        Number of particles to generate.
    r : float
        Radius of the circle.
    rng_seed : int
        Seed for the random number generator for reproducibility.

    Returns
    -------
    x_norm : np.ndarray
        x-coordinates of the generated points.
    px_norm : np.ndarray
        y-coordinates of the generated points.
    '''
    x_norm , px_norm  = generate_hypersphere(num_particles,D=2,r=r, rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm, px_norm


def generate_hypersphere_4D(num_particles,rx =1,ry =1, rng_seed = 0):
    '''
    Generate points uniformly distributed inside a 4-dimensional hypersphere with anisotropic scaling.

    Parameters
    ----------
    num_particles : int
        Number of particles to generate.
    rx : float
        Scaling factor for the x and px dimensions.
    ry : float
        Scaling factor for the y and py dimensions.
    rng_seed : int
        Seed for the random number generator for reproducibility.

    Returns
    -------
    x_norm, px_norm, y_norm, py_norm : np.ndarray
        Coordinates of the generated points in the 4-dimensional space.
    '''

    x_norm , px_norm , y_norm, py_norm = generate_hypersphere(num_particles,D=4,r=[rx,rx,ry,ry], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm


def generate_hypersphere_6D(num_particles,rx =1,ry =1, rzeta=1, rng_seed = 0):
    '''
    Generate points uniformly distributed inside a 6-dimensional hypersphere with anisotropic scaling.

    Parameters
    ----------
    num_particles : int
        Number of particles to generate.
    rx, ry, rzeta : float
        Scaling factors for the x, px, y, py, zeta, and pzeta dimensions respectively.
    rng_seed : int
        Seed for the random number generator for reproducibility.

    Returns
    -------
    x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm : np.ndarray
        Coordinates of the generated points in the 6-dimensional space.
    '''

    x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm = generate_hypersphere(num_particles,D=6,r=[rx,rx,ry,ry,rzeta,rzeta], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm