import numpy as np
import pytest
from scipy.constants import c as clight
from scipy.optimize import root

import xobjects as xo
import xpart as xp
import xtrack as xt
from xpart.longitudinal.generate_longitudinal import get_bucket


def make_line(gamma0, energy_gain, start_turn, rf_position=None):
    circumference = 1000.
    particle_ref = xt.Particles(gamma0=gamma0)
    period = circumference / (float(particle_ref.beta0[0]) * clight)
    end_time = 2000 * period
    p0c = float(particle_ref.p0c[0])
    # A linear momentum ramp has dE0/d(reference_turn) = (dp0c/dt)*C/c.
    program = xt.EnergyProgram(
        t_s=[0., end_time],
        p0c=[p0c, p0c + energy_gain * clight / circumference * end_time])
    momentum_compaction = 1 / 4**2
    if rf_position is not None:
        momentum_compaction = 0.
    eta = momentum_compaction - 1 / gamma0**2
    phase = np.arcsin(energy_gain / 1e5)
    if eta > 0:
        phase = np.pi - phase
    if rf_position is None:
        elements = [xt.LineSegmentMap(
            length=circumference, betx=1., bety=1., qx=.31, qy=.32,
            longitudinal_mode='nonlinear', momentum_compaction_factor=momentum_compaction,
            voltage_rf=1e5, frequency_rf=40 / period, phase_rf=phase)]
    else:
        elements = [xt.Multipole(knl=[0., .002]),
                    xt.Drift(length=200.),
                    xt.Cavity(voltage=0., frequency=40 / period, phase=phase),
                    xt.Drift(length=300.),
                    xt.Multipole(knl=[0., -.002]),
                    xt.Drift(length=300.),
                    xt.Cavity(voltage=0., frequency=40 / period, phase=phase),
                    xt.Drift(length=200.)]
        if rf_position == .2:
            elements[2].voltage = 1e5
        else:
            elements[6].voltage = 1e5
    line = xt.Line(elements=elements,
        particle_ref=particle_ref, energy_program=program)
    t_start = program.get_t_s_at_turn(start_turn)
    line['t_turn_s'] = t_start
    frequency = 40 * float(line.particle_ref.beta0[0]) * clight / circumference
    if rf_position is None:
        line[0].frequency_rf = [frequency]
    else:
        line[2].frequency = frequency
        line[6].frequency = frequency
    line.build_tracker()
    return line, t_start


@pytest.mark.parametrize('gamma0', [3., 5.])
@pytest.mark.parametrize('energy_gain', [-5e4, 5e4])
@pytest.mark.parametrize('start_turn', [0., 37.5])
def test_energy_program_bucket_centre_matches_tracking(gamma0, energy_gain, start_turn):
    line, t_start = make_line(gamma0, energy_gain, start_turn)
    check_bucket_centre(line, t_start, start_turn)


@pytest.mark.parametrize('rf_position', [.2, .8])
@pytest.mark.parametrize('energy_gain', [-5e4, 5e4])
def test_energy_program_bucket_with_cavity(rf_position, energy_gain):
    line, t_start = make_line(3., energy_gain, 37.5, rf_position=rf_position)
    check_bucket_centre(line, t_start, 37.5)


def check_bucket_centre(line, t_start, start_turn):
    bucket = line._get_bucket()
    p0c = float(line.particle_ref.p0c[0])
    program = line.energy_program
    t_end = program.get_t_s_at_turn(start_turn + 1)
    next_p0c = program.get_p0c_at_t_s(t_end)
    line.enable_time_dependent_vars = True

    def one_turn_residual(coordinates):
        p = xt.Particles(p0c=p0c, zeta=coordinates[0], delta=coordinates[1],
                         time_s=t_start, reference_turns=start_turn)
        line.track(p)
        # Compare coordinates at consecutive turn entrances, after applying
        # the next turn's reference update, as the tracker does before RF.
        p.update_p0c_and_energy_deviations(next_p0c)
        return [float(p.zeta[0]) - coordinates[0],
                1000 * (float(p.delta[0]) - coordinates[1])]

    solution = root(one_turn_residual, [float(bucket.z_sfp[0]), bucket.dp0],
                    options={'eps': 1e-8, 'xtol': 1e-10})
    xo.assert_allclose(one_turn_residual(solution.x), 0., atol=1e-10, rtol=0)
    # Bucket matching is first order in delta; compare with the independently
    # solved discrete tracking map rather than duplicating its correction.
    xo.assert_allclose(bucket.dp0, solution.x[1], atol=2e-8, rtol=0)
    xo.assert_allclose(bucket.z_sfp, solution.x[0], atol=1e-3, rtol=0)

    p = xt.Particles(p0c=p0c, zeta=bucket.z_sfp, delta=bucket.dp0,
                     time_s=t_start, reference_turns=start_turn)
    line.track(p, num_turns=300, turn_by_turn_monitor=True)
    xo.assert_allclose(line.record_last_track.zeta, bucket.z_sfp,
                       atol=2e-3 * (bucket.z_right - bucket.z_left), rtol=0)


@pytest.mark.parametrize('sigma_z', [.05, 2.])
def test_energy_program_bunch_uses_bucket_momentum_centre(sigma_z):
    line, t_start = make_line(3., 5e4, 37.5)
    bucket = line._get_bucket()
    np.random.seed(781)
    zeta, delta = xp.generate_longitudinal_coordinates(
        line=line, sigma_z=sigma_z, num_particles=20000)
    xo.assert_allclose(delta.mean(), bucket.dp0,
                       atol=5 * delta.std() / np.sqrt(len(delta)), rtol=0)
    xo.assert_allclose(zeta.std(), sigma_z, rtol=.02)
    assert line['t_turn_s'] == t_start
    assert not line.enable_time_dependent_vars


@pytest.mark.parametrize('energy_gain', [0., 5e4])
def test_energy_program_bucket_momentum_override(energy_gain):
    line, _ = make_line(3., energy_gain, 0.)
    if energy_gain == 0:
        xo.assert_allclose(line._get_bucket().dp0, 0., atol=1e-12, rtol=0)
    assert get_bucket(line, delta0=1e-4).dp0 == 1e-4
