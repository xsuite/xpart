# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()


class DummyCommunicator:
    def __init__(self, n_procs, rank):
        self.n_procs = n_procs
        self.rank = rank

    def Get_size(self):
        return self.n_procs

    def Get_rank(self):
        return self.rank


def test_filling_scheme():
    filename = test_data_folder.joinpath(
        'lhc_no_bb/line_and_particle.json')

    with open(filename, 'r') as fid:
        input_data = json.load(fid)

    line = xt.Line.from_dict(input_data['line'])
    circumference = line.get_length()

    filling_scheme_array = np.zeros(3564)
    n_bunches = 100
    filling_scheme_array[0:n_bunches] = 1
    n_procs = 3
    communicator = DummyCommunicator(n_procs, rank=0)
    h_list = [35640]
    filling_scheme = xp.FillingScheme(filling_scheme_array=filling_scheme_array,
                                      communicator=communicator,
                                      circumference=circumference,
                                      harmonic_list=h_list)

    assert len(filling_scheme.bunches_per_rank[0]) == 34
    assert len(filling_scheme.bunches_per_rank[1]) == 33
    assert len(filling_scheme.bunches_per_rank[2]) == 33

    assert (filling_scheme.bunches_per_rank[0] == np.linspace(0, 33, 34)).all()
    assert (filling_scheme.bunches_per_rank[1] == np.linspace(34, 66, 33)).all()
    assert (filling_scheme.bunches_per_rank[2] == np.linspace(67, 99, 33)).all()
