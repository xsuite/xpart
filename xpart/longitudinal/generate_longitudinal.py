# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo

from .rfbucket_matching import RFBucketMatcher
from .rfbucket_matching import ThermalDistribution
from .rf_bucket import RFBucket
from xtrack.particles import Particles
from .single_rf_harmonic_matcher import SingleRFHarmonicMatcher
from ..general import _print

logger = logging.getLogger(__name__)

def _characterize_line(line, particle_ref,
                          **kwargs # passed to twiss
                          ):

    if line.iscollective:
        logger.warning('Ignoring collective elements in particles generation.')
        line = line._get_non_collective_line()

    T_rev = line.get_length()/(particle_ref._xobject.beta0[0]*clight)
    freq_list = []
    lag_list_deg = []
    voltage_list = []
    h_list = []
    energy_ref_increment_list = []
    found_nonlinear_longitudinal = False
    found_linear_longitudinal = False
    for ee in line.elements:
        if ee.__class__.__name__ == 'Cavity':
            eecp = ee.copy(_context=xo.ContextCpu())
            if ee.voltage != 0:
                freq_list.append(eecp.frequency)
                lag_list_deg.append(eecp.lag)
                voltage_list.append(eecp.voltage)
                h_list.append(eecp.frequency*T_rev)
                found_nonlinear_longitudinal = True
        elif ee.__class__.__name__ == 'LineSegmentMap':
            eecp = ee.copy(_context=xo.ContextCpu())
            assert eecp.longitudinal_mode in [
                'nonlinear', 'linear_fixed_qs', 'linear_fixed_rf', None]
            if eecp.longitudinal_mode in ['nonlinear' , 'linear_fixed_rf']:
                freq_list += list(eecp.frequency_rf)
                lag_list_deg += list(eecp.lag_rf)
                voltage_list += list(eecp.voltage_rf)
                h_list += [ff*T_rev for ff in eecp.frequency_rf]
            if eecp.longitudinal_mode  == 'nonlinear':
                found_nonlinear_longitudinal = True
            elif eecp.longitudinal_mode in ['linear_fixed_qs' , 'linear_fixed_rf']:
                found_linear_longitudinal = True
            if eecp.energy_ref_increment != 0:
                energy_ref_increment_list.append(eecp.energy_ref_increment)
        elif ee.__class__.__name__ == 'ReferenceEnergyIncrease':
            eecp = ee.copy(_context=xo.ContextCpu())
            if eecp.Delta_p0c != 0:
                # valid for small energy change
                # See Wille, The Physics of Particle Accelerators
                # Appendix B, formula B.16 .
                energy_ref_increment_list.append(
                    eecp.Delta_p0c * particle_ref._xobject.beta0[0])


    found_only_linear_longitudinal = False
    if not found_linear_longitudinal and not found_nonlinear_longitudinal:
        raise ValueError('No longitudinal focusing found in the line. '
                         'Cannot generate matched longitudinal coordinates.')

    elif found_linear_longitudinal and found_nonlinear_longitudinal:
        raise ValueError('Generation of matched longitudinal coordinates in line featuring '
                         'both linear and non-linear elements is not implemented')
    else:
        found_only_linear_longitudinal = found_linear_longitudinal

    if found_nonlinear_longitudinal:
        assert len(freq_list) > 0

    tw = line.twiss(
        particle_ref=particle_ref, **kwargs)

    p0c_increase_from_energy_program = None
    if line.energy_program is not None:
        p0c_increase_from_energy_program = line.energy_program.get_p0c_increse_per_turn_at_t_s(
                                                        line.vv['t_turn_s'])


    dct={}
    dct['T_rev'] = T_rev
    dct['freq_list'] = freq_list
    dct['lag_list_deg'] = lag_list_deg
    dct['voltage_list'] = voltage_list
    dct['h_list'] = h_list
    dct['momentum_compaction_factor'] = tw['momentum_compaction_factor']
    dct['slip_factor'] = tw['slip_factor']
    dct['qs'] = tw['qs']
    dct['bets0'] = tw['bets0']
    dct['found_only_linear_longitudinal'] = found_only_linear_longitudinal
    dct['energy_ref_increment_list'] = energy_ref_increment_list
    dct['p0c_increase_from_energy_program'] = p0c_increase_from_energy_program
    return dct

def get_bucket(line, **kwargs):
    kwargs['sigma_z'] = 1.
    return generate_longitudinal_coordinates(line=line,
                                             engine='pyheadtail',
                                             _only_bucket=True,
                                             **kwargs)

def generate_longitudinal_coordinates(
                                    line=None,
                                    num_particles=None,
                                    distribution='gaussian',
                                    sigma_z=None,
                                    engine=None,
                                    return_matcher=False,
                                    particle_ref=None,
                                    mass0=None, q0=None, gamma0=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    energy_ref_increment=None,
                                    tracker=None,
                                    m=None,
                                    q=None,
                                    _only_bucket=False,
                                    **kwargs # passed to twiss
                                    ):

    '''
    Generate longitudinal coordinates matched to given RF parameters (non-linar
    bucket).

    Parameters
    ----------
    line: xline.Line
        Line for which the longitudinal coordinates are generated.
    num_particles: int
        Number of particles to be generated.
    distribution: str
        Distribution of the particles. Possible values are `gaussian` and
        `parabolic` and 'binomial'.
    sigma_z: float
        RMS bunch length in meters.
    engine: str
        Engine to be used for the generation. Possible values are `pyheadtail`
        and `single-rf-harmonic`.
    return_matcher: bool
        If True, the matcher object is returned.
    m : float
        binomial parameter if distribution is 'binomial'
    q : float
        q-Gaussian parameter if distribution is 'qgaussian'

    Returns
    -------
    zeta: np.ndarray
        Longitudinal position of the generated particles.
    delta: np.ndarray
        Longitudinal momentum deviation of the generated particles.
    matcher: object
        Matcher object used for the generation. Returned only if
        `return_matcher` is True.

    '''

    if line is not None and tracker is not None:
        raise ValueError(
            'line and tracker cannot be provided at the same time.')

    if tracker is not None:
        _print('Warning! '
            "The argument tracker is deprecated. Please use line instead.",
            DeprecationWarning)
        line = tracker.line

    if line is not None:
        if particle_ref is None:
            particle_ref = line.particle_ref
        assert particle_ref is not None
        dct = _characterize_line(line, particle_ref, **kwargs)

    assert particle_ref is not None

    if mass0 is None:
        assert particle_ref is not None
        mass0 = particle_ref.mass0

    if q0 is None:
        assert particle_ref is not None
        q0 = particle_ref.q0

    if gamma0 is None:
        assert particle_ref is not None
        gamma0 = particle_ref._xobject.gamma0[0]

    if circumference is None:
        assert line is not None
        circumference = line.get_length()

    if momentum_compaction_factor is None:
        assert line is not None
        momentum_compaction_factor = dct['momentum_compaction_factor']

    if rf_harmonic is None:
        assert line is not None
        rf_harmonic=dct['h_list']

    if rf_voltage is None:
        assert line is not None
        rf_voltage=dct['voltage_list']

    if rf_phase is None:
        assert line is not None
        rf_phase=(np.array(dct['lag_list_deg']))/180*np.pi

    p0c_increase_from_energy_program = 0.
    if energy_ref_increment is None and line is not None:
        energy_ref_increment_list = dct['energy_ref_increment_list']
        if energy_ref_increment_list:
            energy_ref_increment = np.sum(energy_ref_increment_list)
        p0c_increase_from_energy_program = dct['p0c_increase_from_energy_program']

    assert sigma_z is not None

    # Compute beta0 from gamma0
    beta0 = np.sqrt(1 - 1 / gamma0**2)

    matcher = None

    if engine is None:
        if line is not None and dct['found_only_linear_longitudinal']:
            engine = 'linear'
        else:
            engine = 'pyheadtail'

    if engine == "linear":
        if energy_ref_increment is not None and energy_ref_increment != 0:
            raise NotImplementedError(
                'Reference energy increment not yet supported for linear matching')
        if distribution != 'gaussian':
            raise NotImplementedError
        assert line is not None, ('Not yet implemented if line is not provided')
        sigma_dp = sigma_z / np.abs(dct['bets0'])
        z_particles = sigma_z * np.random.normal(size=num_particles)
        delta_particles = sigma_dp * np.random.normal(size=num_particles)
        assert energy_ref_increment is None
    elif engine == "pyheadtail":
        if distribution != 'gaussian':
            raise NotImplementedError

        dp0c_eV = 0.
        if energy_ref_increment:
            dp0c_eV = energy_ref_increment / beta0 # valid for small energy change
                                                # See Wille, The Physics of Particle Accelerators
                                                # Appendix B, formula B.16 .
        if p0c_increase_from_energy_program is not None:
            dp0c_eV += p0c_increase_from_energy_program

        dp0c_J = dp0c_eV * qe
        dp0_si = dp0c_J / clight

        rfbucket = RFBucket(circumference=circumference,
                            gamma=gamma0,
                            mass_kg=mass0/(clight**2)*qe,
                            charge_coulomb=np.abs(q0)*qe,
                            alpha_array=np.atleast_1d(momentum_compaction_factor),
                            harmonic_list=np.atleast_1d(rf_harmonic),
                            voltage_list=np.atleast_1d(rf_voltage),
                            phi_offset_list=np.atleast_1d(rf_phase),
                            p_increment=dp0_si)
        if _only_bucket:
            return rfbucket

        if sigma_z < 0.03 * circumference/np.max(np.atleast_1d(rf_harmonic)):
            logger.info('short bunch, use linear matching')
            if energy_ref_increment is not None and energy_ref_increment != 0:
                raise NotImplementedError(
                    'Reference energy increment not yet supported for linear matching')
            eta = momentum_compaction_factor - 1/particle_ref._xobject.gamma0[0]**2
            beta_z = np.abs(eta) * circumference / 2.0 / np.pi / rfbucket.Q_s
            sigma_dp = sigma_z / beta_z
            z_particles = sigma_z * np.random.normal(size=num_particles)
            delta_particles = sigma_dp * np.random.normal(size=num_particles)
        else:
            matcher = RFBucketMatcher(rfbucket=rfbucket,
                distribution_type=ThermalDistribution,
                sigma_z=sigma_z)
            z_particles, delta_particles, _, _ = matcher.generate(
                                        macroparticlenumber=num_particles)

    elif engine == "single-rf-harmonic":
        if distribution not in ["parabolic", "gaussian", "binomial", "qgaussian"]:
            raise NotImplementedError
        eta = momentum_compaction_factor - 1/particle_ref._xobject.gamma0[0]**2

        # if fragment
        if particle_ref._xobject.chi[0] != 1.0:
            raise NotImplementedError

        sigma_tau = sigma_z/particle_ref._xobject.beta0[0]

        voltage = np.sum(rf_voltage)

        harmonic_number = np.round(rf_harmonic).astype(int)[0]
        if not np.allclose(harmonic_number, rf_harmonic, atol=5.e-1, rtol=0.):
            raise Exception(f"Multiple harmonics detected in lattice: {rf_harmonic}")

        matcher = SingleRFHarmonicMatcher(q0=q0,
                                          voltage=voltage,
                                          length=circumference,
                                          freq=dct['freq_list'][0],
                                          p0c=particle_ref._xobject.p0c[0],
                                          slip_factor=eta,
                                          beta0=particle_ref._xobject.beta0[0],
                                          rms_bunch_length=sigma_tau,
                                          distribution=distribution, m=m, q=q)

        tau, ptau = matcher.sample_tau_ptau(n_particles=num_particles)

        # convert (tau, ptau) to (zeta, delta)
        z_particles = np.array(particle_ref._xobject.beta0[0]) * np.array(tau)  # zeta
        temp_particles = Particles(p0c=particle_ref._xobject.p0c[0],
                                   zeta=z_particles, ptau=ptau)
        delta_particles = np.array(temp_particles.delta)
    else:
        raise NotImplementedError # TODO better message

    if return_matcher:
        return z_particles, delta_particles, matcher
    else:
        return z_particles, delta_particles
