# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import xobjects as xo

from ..general import _pkg_root

from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import m_p


pmass = m_p * clight * clight / qe

LAST_INVALID_STATE = -999999999

size_vars = (
    (xo.Int64, '_capacity'),
    (xo.Int64, '_num_active_particles'),
    (xo.Int64, '_num_lost_particles'),
    (xo.Int64, 'start_tracking_at_element'),
    )
# Capacity is always kept up to date
# the other two are placeholders to be used if needed
# i.e. on ContextCpu

scalar_vars = (
    (xo.Float64, 'q0'),
    (xo.Float64, 'mass0'),
)

part_energy_vars = (
    (xo.Float64, 'ptau'),
    (xo.Float64, 'delta'),
    (xo.Float64, 'rpp'),
    (xo.Float64, 'rvv'),
)

per_particle_vars = (
    (
        (xo.Float64, 'p0c'),
        (xo.Float64, 'gamma0'),
        (xo.Float64, 'beta0'),
        (xo.Float64, 's'),
        # (xo.Float64, 'x'),
        # (xo.Float64, 'y'),
        # (xo.Float64, 'px'),
        # (xo.Float64, 'py'),
        (xo.Float64, 'zeta'),
    )
    + part_energy_vars +
    (
        (xo.Float64, 'chi'),
        (xo.Float64, 'charge_ratio'),
        (xo.Float64, 'weight'),
        (xo.Int64, 'particle_id'),
        (xo.Int64, 'at_element'),
        (xo.Int64, 'at_turn'),
        (xo.Int64, 'state'),
        (xo.Int64, 'parent_particle_id'),
        (xo.UInt32, '_rng_s1'),
        (xo.UInt32, '_rng_s2'),
        (xo.UInt32, '_rng_s3'),
        (xo.UInt32, '_rng_s4')
    )
)


fields = {}
for tt, nn in size_vars + scalar_vars:
    fields[nn] = tt

for tt, nn in per_particle_vars:
    fields[nn] = tt[:]


class ParticlesInterface(xo.HybridClass):
    _cname = 'ParticlesData'
    _xofields = fields
    _extra_c_sources = [
        _pkg_root.joinpath('rng_src', 'base_rng.h'),
        _pkg_root.joinpath('rng_src', 'particles_rng.h'),
        '\n /*placeholder_for_local_particle_src*/ \n'
    ]

    def __init__(self, *args, **kwargs):  # noqa
        raise NotImplementedError('ParticlesInterface is an abstract class to '
                                  'be used as a template for usable concrete '
                                  'implementations. It serves only a purpose of'
                                  'defining the bare minimum C-API. Therefore, '
                                  'it cannot be instantiated.')
