# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import xobjects as xo
import xpart as xp
import xtrack as xt

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

def test_phase_monitor():
    for context in xo.context.get_test_contexts():

        print(f"Test {context.__class__}")

        filename = test_data_folder.joinpath(
            'sps_w_spacecharge/line_no_spacecharge_and_particle.json')
        with open(filename, 'r') as fid:
            ddd = json.load(fid)
        line = xt.Line.from_dict(ddd['line'])
        line.particle_ref = xp.Particles.from_dict(ddd['particle'])

        tracker = xt.Tracker(line=line, _context=context)

        particles = xp.build_particles(tracker=tracker, x_norm=[0.1, 0.2],
                                       y_norm=[0.3, 0.4],
                                       scale_with_transverse_norm_emitt=(2e-6,2e-6),
                                       _context=context)
        phase_monitor = xp.PhaseMonitor(tracker=tracker, num_particles=2,
                                        twiss=tracker.twiss())

        for _ in range(5):
            phase_monitor.measure(particles)
            tracker.track(particles)

        tw = tracker.twiss()
        assert np.allclose(phase_monitor.qx[:], np.mod(tw['qx'], 1),
                           rtol=0, atol=1e-3)
        assert np.allclose(phase_monitor.qy[:], np.mod(tw['qy'], 1),
                           rtol=0, atol=1e-3)

