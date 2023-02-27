# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import json

import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_classical_particle_radius_ions(test_context):
    """
    Test classical particle radius for ions
    """
    # Load SPS ion sequence with Pb82 ions
    fname_line = test_data_folder.joinpath('sps_ions/line_and_particle.json')
    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)

    particle_ref = xp.Particles.from_dict(input_data['particle_ref'],
                                          _context=test_context)
    r0_Pb82 = np.float64(4.998617e-17)  # calculated theoretical classical particle radius

    assert np.isclose(particle_ref.get_classical_particle_radius0(), r0_Pb82, atol=1e-5)


@for_all_test_contexts
def test_classical_particle_radius_protons(test_context):
    """
    Test classical particle radius for protons
    """
    # Load SPS sequence without space charge
    fname_line = test_data_folder.joinpath('sps_w_spacecharge/'
                                           'line_no_spacecharge_and_particle.json')
    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)

    particle_ref = xp.Particles.from_dict(input_data['particle'],
                                          _context=test_context)
    r0_proton = np.float64(1.534698e-18)  # calculated theoretical classical particle radius

    assert np.isclose(particle_ref.get_classical_particle_radius0(), r0_proton, atol=1e-5)


def test_showing():
    """
    Test whether showing particle values gives reasonable output
    """
    test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

    # Load SPS ion sequence with Pb82 ions
    fname_line = test_data_folder.joinpath('sps_ions/line_and_particle.json')
    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)

    particle_ref = xp.Particles.from_dict(input_data['particle_ref'])
    particle_ref.show()
