import numpy as np


class FillingScheme:
    def __init__(self, bunch_spacing_in_buckets, filling_scheme_array, harmonic_list, circumference, communicator=None):
        self.bucket_length = circumference / (2*np.amin(harmonic_list))
        self.bunch_spacing = bunch_spacing_in_buckets*self.bucket_length
        self.filling_scheme_array = filling_scheme_array
        self._communicator = communicator
        self.bunch_ids = np.unique(np.cumsum(filling_scheme_array == 1)) - 1
        self.n_bunches = len(self.bunch_ids)
        self.bunches_per_rank = []
        self.split_scheme()

        self.n_bunches = int(np.sum(filling_scheme_array))
        # I need this map when generating the beam to know in which bucket to place the particles. Maybe we could use
        # the slicer here..
        self.bunch_id_bucket_id_map = np.zeros(self.n_bunches)

        bunch_id = 0
        for bucket_id, bucket_full in enumerate(filling_scheme_array):
            if bucket_full:
                self.bunch_id_bucket_id_map[bunch_id] = bucket_id
                bunch_id += 1

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
            self.bunches_per_rank = [np.linspace(0, self.n_bunches-1, self.n_bunches)]

    def get_bunches(self):
        return self.bunches_per_rank[self._communicator.Get_rank()]
