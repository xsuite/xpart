import xobjects as xo
import xpart as xp
import xtrack as xt
# import xcoll as xc

from ..general import _pkg_root

import numpy as np



class RandomGenerator(xo.HybridClass):
    _xofields = {
        'rutherford_iterations':    xo.Int8,
        'rutherford_lower_val':     xo.Float64,
        'rutherford_upper_val':     xo.Float64,
        'rutherford_A':             xo.Float64,
        'rutherford_B':             xo.Float64
    }

    _depends_on = [xp.Particles]

    _extra_c_sources = [
        _pkg_root.joinpath('random_number_generator','rng_src','exponential_integral_Ei.h'),
        _pkg_root.joinpath('random_number_generator','rng_src','random_generator.h')
    ]

    _kernels = {
        'set_rutherford': xo.Kernel(
            c_name='RandomGeneratorData_set_rutherford',
            args=[
                xo.Arg(xo.ThisClass, name='ran'),
                xo.Arg(xo.Float64, name='z'),
                xo.Arg(xo.Float64, name='emr'),
                xo.Arg(xo.Float64, name='upper_val')
            ])
        }

    def __init__(self, **kwargs):
        if '_xobject' not in kwargs:
            kwargs.setdefault('rutherford_iterations', 7)
            kwargs.setdefault('rutherford_lower_val', 0.0009982)
            kwargs.setdefault('rutherford_upper_val', 1.)
            kwargs.setdefault('rutherford_A', 1.)
            kwargs.setdefault('rutherford_B', 1.)
            kwargs.setdefault('_samples', np.array([]))
        super().__init__(**kwargs)

    def set_rutherford_by_xcoll_material(self, material):
#         assert isinstance(material, xc.GeneralMaterial)
        self.compile_kernels(only_if_needed=True, save_source_as='randomtest.c')
        context = self._buffer.context
        context.kernels.set_rutherford(ran=self, z=material.Z, emr=material.nuclear_radius, upper_val=material.hcut)



class RandomSampler(xt.BeamElement):
    _xofields = {
        'distribution': xo.Int8,
        'generator':    RandomGenerator,
        'n_samples':    xo.Int64,
        '_samples':     xo.Float64[:]
    }

    distributions = {
        'uniform':     0,
        'exponential': 1,
        'gaussian':    2,
        'rutherford':  3
    }

    _extra_c_sources = [
        _pkg_root.joinpath('random_number_generator','rng_src','base_rng.h'),
        _pkg_root.joinpath('random_number_generator','rng_src','local_particle_rng.h'),
        _pkg_root.joinpath('random_number_generator','rng_src','random_sampler.h')
    ]

    def __init__(self, **kwargs):
        if '_xobject' not in kwargs:
            kwargs.setdefault('distribution', 0)
            kwargs.setdefault('generator', RandomGenerator())
            kwargs.setdefault('n_samples', 1000)
            n_seeds = kwargs.pop('n_seeds', 1000)
            capacity = int(max(1e6, kwargs['n_samples']*n_seeds))
            kwargs.setdefault('_samples', np.zeros(capacity))
        super().__init__(**kwargs)


    def sample(self, n_samples=1000, n_seeds=None, distribution='uniform', particles=None):
        if distribution not in self.distributions.keys():
            raise valueError(f"The variable 'distribution' should be one of {self.distributions.keys()}.")
        self.distribution = self.distributions[distribution]

        if particles is None:
            if n_seeds is None:
                n_seeds = 1000
            particles = xp.Particles(_capacity=n_seeds)
            particles._init_random_number_generator()
        elif n_seeds is not None and n_seeds != len(particles._rng_s1):
            print("Warning: both 'particles' and 'n_seeds' are given, but are not compatible. Ignoring 'n_seeds'...")
        if len(particles._rng_s1) > len(self._samples):
            raise ValueError("More seeds requested than capacity in RandomSampler._samples.")

        if n_samples*len(particles._rng_s1) > len(self._samples):
            # Todo: resize _samples
            print("Not enough capacity in RandomSampler._samples to allow for requested number of seeds and samples. "
                  + "Downsized 'n_samples' to accomodate.")
            n_samples = int( len(self._samples) / len(particles._rng_s1) )

        self.n_samples = n_samples
        self.track(particles)
        return self._samples, particles



