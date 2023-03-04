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
def test_gaussian_bunch_generation(test_context):
    bunch_intensity = 1e11
    sigma_z = 22.5e-2
    n_part = int(5e6)
    nemitt_x = 2e-6
    nemitt_y = 2.5e-6

    filename = test_data_folder.joinpath(
        'sps_w_spacecharge/line_no_spacecharge_and_particle.json')
    with open(filename, 'r') as fid:
        ddd = json.load(fid)
    line = xt.Line.from_dict(ddd['line'])
    line.particle_ref = xp.Particles.from_dict(ddd['particle'],
                           _context=test_context # for testing purposes
                           )

    line.build_tracker(_context=test_context)

    part_on_co = line.find_closed_orbit()

    part = xp.generate_matched_gaussian_bunch(
             _context=test_context,
             line=line,
             num_particles=n_part,
             total_intensity_particles=bunch_intensity,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z)

    tw = line.twiss()

    # CHECKS
    y_rms = np.std(test_context.nparray_from_context_array(part.y))
    py_rms = np.std(test_context.nparray_from_context_array(part.py))
    x_rms = np.std(test_context.nparray_from_context_array(part.x))
    px_rms = np.std(test_context.nparray_from_context_array(part.px))
    delta_rms = np.std(test_context.nparray_from_context_array(part.delta))
    zeta_rms = np.std(test_context.nparray_from_context_array(part.zeta))

    part_on_co.move(_context=xo.ContextCpu())
    gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
    gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
    assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
    assert np.isclose(x_rms,
                      np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
                      rtol=1e-2, atol=1e-15)
    assert np.isclose(y_rms,
                      np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
                      rtol=1e-2, atol=1e-15)


@for_all_test_contexts
def test_short_bunch(test_context):
    bunch_intensity = 1e11
    sigma_z = 22.5e-2/5
    n_part = int(5e3)
    nemitt_x = 2e-6
    nemitt_y = 2.5e-6

    filename = test_data_folder.joinpath(
            'sps_w_spacecharge/line_no_spacecharge_and_particle.json')
    with open(filename, 'r') as fid:
        ddd = json.load(fid)
    line = xt.Line.from_dict(ddd['line'])
    particle_ref = xp.Particles.from_dict(ddd['particle'])
    line.build_tracker(_context=test_context)

    tw = line.twiss(particle_ref=particle_ref)

    part = xp.generate_matched_gaussian_bunch(
             _context=test_context,
             num_particles=n_part, total_intensity_particles=bunch_intensity,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
             particle_ref=particle_ref, line=line)

    # CHECKS
    y_rms = np.std(test_context.nparray_from_context_array(part.y))
    py_rms = np.std(test_context.nparray_from_context_array(part.py))
    x_rms = np.std(test_context.nparray_from_context_array(part.x))
    px_rms = np.std(test_context.nparray_from_context_array(part.px))
    delta_rms = np.std(test_context.nparray_from_context_array(part.delta))
    zeta_rms = np.std(test_context.nparray_from_context_array(part.zeta))

    gemitt_x = nemitt_x/particle_ref.beta0[0]/particle_ref.gamma0[0]
    gemitt_y = nemitt_y/particle_ref.beta0[0]/particle_ref.gamma0[0]
    assert np.isclose(zeta_rms, sigma_z, rtol=5e-2, atol=1e-15)
    assert np.isclose(x_rms,
                      np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
                      rtol=5e-2, atol=1e-15)
    assert np.isclose(y_rms,
                      np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
                      rtol=5e-2, atol=1e-15)

    for iturn in range(100):
        line.track(part)
        assert np.isclose(zeta_rms, sigma_z, rtol=5e-2, atol=1e-15)
