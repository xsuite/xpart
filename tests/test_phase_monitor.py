# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_phase_monitor(test_context):
    filename = test_data_folder.joinpath(
        'sps_w_spacecharge/line_no_spacecharge_and_particle.json')
    with open(filename, 'r') as fid:
        ddd = json.load(fid)
    line = xt.Line.from_dict(ddd['line'])
    line.particle_ref = xp.Particles.from_dict(ddd['particle'])

    line.build_tracker(_context=test_context)

    particles = xp.build_particles(line=line, x_norm=[0.1, 0.2],
                                   y_norm=[0.3, 0.4],
                                   nemitt_x=2e-6, nemitt_y=2e-6,
                                   _context=test_context)
    phase_monitor = xp.PhaseMonitor(line=line, num_particles=2,
                                    twiss=line.twiss())

    for _ in range(5):
        phase_monitor.measure(particles)
        line.track(particles)

    tw = line.twiss()
    assert np.allclose(phase_monitor.qx[:], np.mod(tw['qx'], 1),
                       rtol=0, atol=1e-3)
    assert np.allclose(phase_monitor.qy[:], np.mod(tw['qy'], 1),
                       rtol=0, atol=1e-3)

