
import numpy as np



# Adapted from : https://baezortega.github.io/2018/10/14/hypersphere-sampling/
def hypersphere(N, D, r=1, rng_seed = 0, surface=False ,unpack = False):
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
    


def hypersphere_2D(num_particles,r = 1, rng_seed = 0):
    
    x_norm , px_norm  = hypersphere(num_particles,D=2,r=r, rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm, px_norm


def hypersphere_4D(num_particles,rx =1,ry =1, rng_seed = 0):
    
    x_norm , px_norm , y_norm, py_norm = hypersphere(num_particles,D=4,r=[rx,rx,ry,ry], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm


def hypersphere_6D(num_particles,rx =1,ry =1, rzeta=1, rng_seed = 0):
    
    x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm = hypersphere(num_particles,D=6,r=[rx,rx,ry,ry,rzeta,rzeta], rng_seed=rng_seed, surface = False,unpack=True)

    return x_norm , px_norm , y_norm, py_norm, zeta_norm, pzeta_norm