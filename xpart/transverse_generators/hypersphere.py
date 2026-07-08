
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
        new_radii = rng.uniform(low=0.0, high=1.0, size=(N, 1)) ** (1 / D)
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
    """
    Generate points uniformly distributed inside a 2D disk.

    The returned coordinates are normalized coordinates sampled uniformly in
    area inside a disk of radius `r`.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.
    r : float, optional
        Radius of the disk.
    rng_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    x_norm : np.ndarray
        First normalized coordinate.
    px_norm : np.ndarray
        Second normalized coordinate.

    Example
    -------

    .. code-block:: python

        import xpart as xp

        x_norm, px_norm = xp.generate_hypersphere_2D(
            num_particles=4,
            r=2.0,
            rng_seed=12345)

        x_norm  # [-1.226886, -1.860261, -0.100816, -1.760178]
        px_norm # [1.088933, -0.553751, -0.991372, 0.835044]
    """
    x_norm , px_norm  = generate_hypersphere(num_particles,D=2,r=r, rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm, px_norm


def generate_hypersphere_4D(num_particles,rx =1,ry =1, rng_seed = 0):
    """
    Generate points uniformly distributed inside a scaled 4D hypersphere.

    The generated coordinates satisfy an ellipsoidal scaling where the `x` and
    `px` axes use radius `rx`, and the `y` and `py` axes use radius `ry`.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.
    rx : float, optional
        Radius used for the `x_norm` and `px_norm` coordinates.
    ry : float, optional
        Radius used for the `y_norm` and `py_norm` coordinates.
    rng_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    x_norm : np.ndarray
        First horizontal normalized coordinate.
    px_norm : np.ndarray
        Second horizontal normalized coordinate.
    y_norm : np.ndarray
        First vertical normalized coordinate.
    py_norm : np.ndarray
        Second vertical normalized coordinate.

    Example
    -------

    .. code-block:: python

        import xpart as xp

        x_norm, px_norm, y_norm, py_norm = xp.generate_hypersphere_4D(
            num_particles=4,
            rx=2.0,
            ry=3.0,
            rng_seed=12345)

        x_norm  # [-1.233699, -0.067512, 0.207331, -0.819284]
        px_norm # [1.094981, -0.663875, -1.121394, 0.97336]
        y_norm  # [-1.131601, -1.838431, 2.021932, -0.755677]
        py_norm # [-0.336848, 0.872168, 0.834211, -0.098215]
    """

    x_norm , px_norm , y_norm, py_norm = generate_hypersphere(num_particles,D=4,r=[rx,rx,ry,ry], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm


def generate_hypersphere_6D(num_particles,rx =1,ry =1, rzeta=1, rng_seed = 0):
    """
    Generate points uniformly distributed inside a scaled 6D hypersphere.

    The generated coordinates satisfy an ellipsoidal scaling where the `x` and
    `px` axes use radius `rx`, the `y` and `py` axes use radius `ry`, and the
    `zeta` and `pzeta` axes use radius `rzeta`.

    Parameters
    ----------
    num_particles : int
        Number of points to generate.
    rx : float, optional
        Radius used for the `x_norm` and `px_norm` coordinates.
    ry : float, optional
        Radius used for the `y_norm` and `py_norm` coordinates.
    rzeta : float, optional
        Radius used for the `zeta_norm` and `pzeta_norm` coordinates.
    rng_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    x_norm : np.ndarray
        First horizontal normalized coordinate.
    px_norm : np.ndarray
        Second horizontal normalized coordinate.
    y_norm : np.ndarray
        First vertical normalized coordinate.
    py_norm : np.ndarray
        Second vertical normalized coordinate.
    zeta_norm : np.ndarray
        First longitudinal normalized coordinate.
    pzeta_norm : np.ndarray
        Second longitudinal normalized coordinate.

    Example
    -------

    .. code-block:: python

        import xpart as xp

        x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm = (
            xp.generate_hypersphere_6D(
                num_particles=4,
                rx=2.0,
                ry=3.0,
                rzeta=1.5,
                rng_seed=12345)
        )

        x_norm     # [-1.02114, -0.742471, -0.591355, 0.297695]
        px_norm    # [0.906322, 0.352235, 0.702565, 0.723216]
        y_norm     # [-0.936633, 0.293987, -0.545443, 1.025363]
        py_norm    # [-0.278811, -1.590091, -0.070891, -0.232398]
        zeta_norm  # [-0.040526, 0.955673, 0.46072, 0.35008]
        pzeta_norm # [-0.398511, 0.394293, -0.73395, -0.62872]
    """

    x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm = generate_hypersphere(num_particles,D=6,r=[rx,rx,ry,ry,rzeta,rzeta], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm
