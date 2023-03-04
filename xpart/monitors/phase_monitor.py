# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

class PhaseMonitor:
    def __init__(self, tracker=None, num_particles=1, twiss=None, line=None):

        if line is not None:
            assert tracker is None
            assert line.tracker is not None
            tracker = line.tracker

        self.twiss = twiss
        self.phase_x = []
        self.phase_y = []
        import xtrack as xt # Avoid circular dependency between xtrack as xpart
        self._monitor = xt.ParticlesMonitor(_context=tracker._buffer.context,
                                         num_particles=num_particles,
                                         start_at_turn=0,
                                         stop_at_turn=1)


    def measure(self, particles):
        self._monitor.track(particles)

        delta = self._monitor.delta
        tw = self.twiss

        for ss in 'xy':
            r = getattr(self._monitor, ss).copy()
            pr = getattr(self._monitor, f'p{ss}').copy()

            r -= delta*tw[f'd{ss}'][0]
            pr -= delta*tw[f'dp{ss}'][0]

            betr = tw[f'bet{ss}'][0]
            alfr = tw[f'alf{ss}'][0]

            phase = np.angle(r / np.sqrt(betr) -
                       1j*(r * alfr / np.sqrt(betr) +
                       pr * np.sqrt(betr)))

            getattr(self, f'phase_{ss}').append(
                              np.atleast_1d(np.squeeze(phase)))

        self._monitor.start_at_turn += 1
        self._monitor.stop_at_turn += 1

    @property
    def qx(self):
        return np.mod(np.diff(np.array(self.phase_x), axis=0)/(2*np.pi), 1)

    @property
    def qy(self):
        return np.mod(np.diff(np.array(self.phase_y), axis=0)/(2*np.pi), 1)
