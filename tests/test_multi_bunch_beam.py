import xpart as xp
import numpy as np


class DummyCommunicator:
    def __init__(self, n_procs):
        self.n_procs = n_procs

    def Get_size(self):
        return self.n_procs


def test_filling_scheme():
    bunch_spacing_in_buckets = 10
    filling_scheme_array = np.zeros(3564)
    filling_scheme_array[0:100] = 1
    n_bunches = 100
    filled_slots = np.linspace(0, n_bunches, n_bunches)
    n_procs = 3
    communicator = DummyCommunicator(n_procs)
    filling_scheme = xp.FillingScheme(bunch_spacing_in_buckets, filling_scheme_array, communicator)

    assert (filling_scheme.bunches_per_rank[0] == np.linspace(0, 33, 34)).all()
    assert (filling_scheme.bunches_per_rank[1] == np.linspace(34, 66, 33)).all()
    assert (filling_scheme.bunches_per_rank[2] == np.linspace(67, 99, 33)).all()
