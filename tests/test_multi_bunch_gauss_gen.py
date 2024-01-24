# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts
test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()


class DummyCommunicator:
    def __init__(self, n_procs, rank):
        self.n_procs = n_procs
        self.rank = rank

    def Get_size(self):
        return self.n_procs

    def Get_rank(self):
        return self.rank


@for_all_test_contexts
def test_multi_bunch_gaussian_generation(test_context):
    bunch_intensity = 1e11
    sigma_z = 22.5e-2 / 5
    n_part_per_bunch = int(1e5)
    nemitt_x = 2e-6
    nemitt_y = 2.5e-6

    filename = test_data_folder.joinpath(
        'lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(
        input_data['particle'],
        _context=test_context  # for testing purposes
    )

    line.build_tracker(_context=test_context)

    part_on_co = line.find_closed_orbit()

    circumference = line.get_length()
    h_list = [35640]
    bunch_spacing_in_buckets = 10
    filling_scheme = np.zeros(int(np.amin(h_list)/bunch_spacing_in_buckets))
    # build a dummy filling scheme
    n_bunches_tot = 10
    filling_scheme[0:int(n_bunches_tot/2)] = 1
    filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1

    # build a dummy communicator made of two ranks and use rank 0
    n_procs = 2
    rank = 0
    communicator = DummyCommunicator(n_procs, rank)

    first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                             communicator=communicator)

    part = xp.generate_matched_gaussian_multibunch_beam(
        _context=test_context,
        filling_scheme=filling_scheme,  # engine='linear',
        num_particles=n_part_per_bunch,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=10,
        i_bunch_0=first_bunch, num_bunches=n_bunches,
        particle_ref=line.particle_ref
    )

    tw = line.twiss()

    # CHECKS
    for i_bunch in range(n_bunches):
        y_rms = np.std(
            test_context.nparray_from_context_array(
                part.y[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        x_rms = np.std(
            test_context.nparray_from_context_array(
                part.x[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        delta_rms = np.std(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                           (i_bunch+1)*n_part_per_bunch]))
        zeta_rms = np.std(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                          (i_bunch+1)*n_part_per_bunch]))

        part_on_co.move(_context=xo.ContextCpu())

        gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
        gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(
            x_rms,
            np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15)
        assert np.isclose(
            y_rms,
            np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15)

    # build a dummy communicator made of two ranks and use rank 1
    rank = 1
    communicator = DummyCommunicator(n_procs, rank)

    first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                             communicator=communicator)

    part = xp.generate_matched_gaussian_multibunch_beam(
        filling_scheme=filling_scheme,  # engine='linear',
        num_particles=n_part_per_bunch,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=10,
        i_bunch_0=first_bunch, num_bunches=n_bunches,
        particle_ref=line.particle_ref
    )

    # CHECKS
    for i_bunch in range(n_bunches):
        y_rms = np.std(
            test_context.nparray_from_context_array(
                part.y[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        x_rms = np.std(
            test_context.nparray_from_context_array(
                part.x[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        delta_rms = np.std(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                           (i_bunch+1)*n_part_per_bunch]))
        zeta_rms = np.std(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                          (i_bunch+1)*n_part_per_bunch]))

        part_on_co.move(_context=xo.ContextCpu())

        gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
        gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(
            x_rms,
            np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15
        )
        assert np.isclose(
            y_rms,
            np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15
        )

