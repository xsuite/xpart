import json

import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p

import xobjects as xo
import xpart as xp
import xtrack as xt

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

def test_gaussian_bunch_generation():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        bunch_intensity = 1e11
        sigma_z = 22.5e-2
        n_part = int(5e6)
        nemitt_x = 2e-6
        nemitt_y = 2.5e-6

        filename = test_data_folder.joinpath(
                'sps_w_spacecharge/optics_and_co_at_start_ring.json')
        with open(filename, 'r') as fid:
            ddd = json.load(fid)
        RR = np.array(ddd['RR_madx'])
        part_on_co = xp.Particles.from_dict(ddd['particle_on_madx_co'])

        part = xp.generate_matched_gaussian_bunch(
                 _context=context,
                 num_particles=n_part, total_intensity_particles=bunch_intensity,
                 nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                 particle_on_co=part_on_co, R_matrix=RR,
                 circumference=6911., momentum_compaction_factor=0.0030777,
                 rf_harmonic=4620, rf_voltage=3e6, rf_phase=0)

        # CHECKS
        y_rms = np.std(context.nparray_from_context_array(part.y))
        py_rms = np.std(context.nparray_from_context_array(part.py))
        x_rms = np.std(context.nparray_from_context_array(part.x))
        px_rms = np.std(context.nparray_from_context_array(part.px))
        delta_rms = np.std(context.nparray_from_context_array(part.delta))
        zeta_rms = np.std(context.nparray_from_context_array(part.zeta))

        gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
        gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(x_rms,
                     np.sqrt(ddd['betx']*gemitt_x + ddd['dx']**2*delta_rms**2),
                     rtol=1e-2, atol=1e-15)
        assert np.isclose(y_rms,
                     np.sqrt(ddd['bety']*gemitt_y + ddd['dy']**2*delta_rms**2),
                     rtol=1e-2, atol=1e-15)

def test_short_bunch():
       for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

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
        tracker = xt.Tracker(_context=context, line=line)

        tw = tracker.twiss(particle_ref=particle_ref)

        part = xp.generate_matched_gaussian_bunch(
                 _context=context,
                 num_particles=n_part, total_intensity_particles=bunch_intensity,
                 nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                 particle_ref=particle_ref, tracker=tracker)

        # CHECKS
        y_rms = np.std(context.nparray_from_context_array(part.y))
        py_rms = np.std(context.nparray_from_context_array(part.py))
        x_rms = np.std(context.nparray_from_context_array(part.x))
        px_rms = np.std(context.nparray_from_context_array(part.px))
        delta_rms = np.std(context.nparray_from_context_array(part.delta))
        zeta_rms = np.std(context.nparray_from_context_array(part.zeta))

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
            tracker.track(part)
            assert np.isclose(zeta_rms, sigma_z, rtol=5e-2, atol=1e-15)
