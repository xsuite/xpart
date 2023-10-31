import numpy as np


class FillingScheme:
    def __init__(self, bunch_spacing_in_buckets, filling_scheme_array, communicator=None):
        self.bunch_spacing_in_buckets = bunch_spacing_in_buckets
        self.filling_scheme_file = filling_scheme_array
        self._communicator = communicator
        self.bunch_ids = np.unique(np.cumsum(filling_scheme_array == 1)) - 1
        self.n_bunches = len(self.bunch_ids)
        self.bunches_per_rank = []
        self.split_scheme()

    def split_scheme(self):
        if self._communicator is not None:
            n_procs = self._communicator.Get_size()

            # create the array containing the id of the bunches on each rank (copied from PyHEADTAIL.mpi.mpi_data)
            n_bunches = self.n_bunches
            n_bunches_on_rank = [n_bunches // n_procs + 1 if i < n_bunches % n_procs else
                                 n_bunches // n_procs + 0 for i in range(n_procs)]
            print(n_bunches)
            n_tasks_cumsum = np.insert(np.cumsum(n_bunches_on_rank), 0, 0)
            self.bunches_per_rank = [self.bunch_ids[n_tasks_cumsum[i]:n_tasks_cumsum[i + 1]] for i in range(n_procs)]
        else:
            self.bunches_per_rank = np.linspace(0, self.n_bunches-1, self.n_bunches)
