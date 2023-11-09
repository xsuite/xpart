# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np


class FillingScheme:
    """
    A class representing a filling scheme. When a communicator is passed at initialization the filling scheme is
    distributed among the processors and the class holds the information of which bunches belong to which processor
    (in the variable `bunches_per_rank`)
    """
    def __init__(self, filling_scheme_array, harmonic_list, circumference, communicator=None):
        """
        The initializer of the `FillingScheme` class
        :param filling_scheme_array: an array holding for each bunch slot a 0 if the slot is empty and a 1 if the slot
        is full
        :param harmonic_list: the list of the RF harmonics in the machine (the lowest is used to determine the RF
        bucket length)
        :param circumference: the machine circumference in meters
        :param communicator: (optional) an MPI communicator used to distribute the filling scheme
        """
        self.bucket_length = circumference / (np.amin(harmonic_list))
        bunch_spacing_in_buckets = np.amin(harmonic_list)/len(filling_scheme_array)
        self.bunch_spacing = bunch_spacing_in_buckets*self.bucket_length
        self.filling_scheme_array = filling_scheme_array
        self._communicator = communicator
        self.total_bunch_ids = np.unique(np.cumsum(filling_scheme_array == 1)) - 1
        self.total_n_bunches = len(self.total_bunch_ids)
        self.bunches_per_rank = []
        self.split_scheme()

        self.total_n_bunches = int(np.sum(filling_scheme_array))
        # I need this map when generating the beam to know in which bucket to place the particles. Maybe we could use
        # the slicer here..
        self.bunch_id_bucket_id_map = np.zeros(self.total_n_bunches)

        bunch_id = 0
        for bucket_id, bucket_full in enumerate(filling_scheme_array):
            if bucket_full:
                self.bunch_id_bucket_id_map[bunch_id] = bucket_id
                bunch_id += 1

    def split_scheme(self):
        """
        Distribute the filling scheme between the processes, i.e. assign to each processor its bunches
        """
        if self._communicator is not None:
            n_procs = self._communicator.Get_size()

            # create the array containing the id of the bunches on each rank (copied from PyHEADTAIL.mpi.mpi_data)
            n_bunches = self.total_n_bunches
            n_bunches_on_rank = [n_bunches // n_procs + 1 if i < n_bunches % n_procs else
                                 n_bunches // n_procs + 0 for i in range(n_procs)]
            print(n_bunches)
            n_tasks_cumsum = np.insert(np.cumsum(n_bunches_on_rank), 0, 0)
            self.bunches_per_rank = [self.total_bunch_ids[n_tasks_cumsum[i]:n_tasks_cumsum[i + 1]] for i in range(n_procs)]
        else:
            self.bunches_per_rank = [np.linspace(0, self.total_n_bunches - 1, self.total_n_bunches)]

    def get_bunches(self):
        """
        Return the list of bunches assigned to the current rank (i.e. to the rank returned by communicator.GetRank())
        """
        return self.bunches_per_rank[self._communicator.Get_rank()]
