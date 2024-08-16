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
    tw = line.twiss()
    sigma_delta = sigma_z / tw['bets0']
    circumference = line.get_length()
    h_list = [35640]
    bunch_spacing_in_buckets = 10
    bucket_length = circumference/h_list[0]
    bunch_spacing = bunch_spacing_in_buckets * bucket_length
    filling_scheme = np.zeros(int(np.amin(h_list)/bunch_spacing_in_buckets))
    # build a dummy filling scheme
    n_bunches_tot = 10
    filling_scheme[0:int(n_bunches_tot/2)] = 1
    filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1
    filled_slots = filling_scheme.nonzero()[0]

    # make a test faking 3 procs sharing the bunches
    n_procs = 3

    bunch_selection_per_rank = xp.matched_gaussian.split_scheme(filling_scheme=filling_scheme,
                                            n_chunk=n_procs)


    expected_bunch_selection_per_rank = {
        0: np.array([0, 1, 2, 3]),
        1: np.array([4, 5, 6]),
        2: np.array([7, 8, 9])
    }

    for rank in range(n_procs):
        xo.assert_allclose(bunch_selection_per_rank[rank],
                           expected_bunch_selection_per_rank[rank],
                           atol=0, rtol=1e-15)
        part = xp.generate_matched_gaussian_multibunch_beam(
            _context=test_context,
            filling_scheme=filling_scheme,
            bunch_num_particles=n_part_per_bunch,
            bunch_intensity_particles=bunch_intensity,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
            line=line, bunch_spacing_buckets=bunch_spacing_in_buckets,
            bunch_selection=bunch_selection_per_rank[rank],
            particle_ref=line.particle_ref
        )

        assert len(part.x) == n_part_per_bunch*len(bunch_selection_per_rank[rank])

        for i_bunch,bunch_number in enumerate(bunch_selection_per_rank[rank]):
            zeta_avg = np.average(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
            delta_avg = np.average(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
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

            assert np.isclose((zeta_avg+bunch_spacing*filled_slots[bunch_number])/sigma_z, 0.0, atol=1e-2)
            assert np.isclose(delta_avg/sigma_delta, 0.0, atol=1e-2)
            assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
            assert np.isclose(delta_rms, sigma_delta, rtol=1e-1, atol=1e-15)

            part_on_co.move(_context=xo.ContextCpu(omp_num_threads=0))

            gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
            gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
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

            assert np.isclose(
                np.sum(test_context.nparray_from_context_array(
                part.weight[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch])),
                bunch_intensity,
                rtol=1e-2, atol=1e-15
            )

