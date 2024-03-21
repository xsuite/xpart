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
    return dct

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
                                    p_increment=0.,
                                    tracker=None,
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
        `parabolic`.
    sigma_z: float
        RMS bunch length in meters.
    engine: str
        Engine to be used for the generation. Possible values are `pyheadtail`
        and `single-rf-harmonic`.
    return_matcher: bool
        If True, the matcher object is returned.

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
        rf_phase=(np.array(dct['lag_list_deg']) - 180)/180*np.pi

    assert sigma_z is not None

    if engine is None:
        if line is not None and dct['found_only_linear_longitudinal']:
            engine = 'linear'
        else:
            engine = 'pyheadtail'

    if engine == "linear":
        if distribution != 'gaussian':
            raise NotImplementedError
        assert line is not None, ('Not yet implemented if line is not provided')
        sigma_dp = sigma_z / np.abs(dct['bets0'])
        z_particles = sigma_z * np.random.normal(size=num_particles)
        delta_particles = sigma_dp * np.random.normal(size=num_particles)
    elif engine == "pyheadtail":
        if distribution != 'gaussian':
            raise NotImplementedError

        rfbucket = RFBucket(circumference=circumference,
                            gamma=gamma0,
                            mass_kg=mass0/(clight**2)*qe,
                            charge_coulomb=np.abs(q0)*qe,
                            alpha_array=np.atleast_1d(momentum_compaction_factor),
                            harmonic_list=np.atleast_1d(rf_harmonic),
                            voltage_list=np.atleast_1d(rf_voltage),
                            phi_offset_list=np.atleast_1d(rf_phase),
                            p_increment=p_increment)

        if sigma_z < 0.03 * circumference/np.max(np.atleast_1d(rf_harmonic)):
            logger.info('short bunch, use linear matching')
            eta = momentum_compaction_factor - 1/particle_ref._xobject.gamma0[0]**2
            beta_z = np.abs(eta) * circumference / 2.0 / np.pi / rfbucket.Q_s
            sigma_dp = sigma_z / beta_z

            z_particles = sigma_z * np.random.normal(size=num_particles)
            delta_particles = sigma_dp * np.random.normal(size=num_particles)

        else:
            # Generate longitudinal coordinates
            matcher = RFBucketMatcher(rfbucket=rfbucket,
                distribution_type=ThermalDistribution,
                sigma_z=sigma_z)
            z_particles, delta_particles, _, _ = matcher.generate(
                                                    macroparticlenumber=num_particles)
    elif engine == "single-rf-harmonic":
        if distribution not in ["parabolic", "gaussian"]:
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
                                          distribution=distribution)

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
