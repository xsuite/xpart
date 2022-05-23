import logging

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo

from .rfbucket_matching import RFBucketMatcher
from .rfbucket_matching import ThermalDistribution
from .rf_bucket import RFBucket

logger = logging.getLogger(__name__)

def _characterize_tracker(tracker, particle_ref):

    if tracker.iscollective:
        logger.warning('Ignoring collective elements in particles generation.')
        tracker = tracker._supertracker

    line = tracker.line
    T_rev = line.get_length()/(particle_ref._xobject.beta0[0]*clight)
    freq_list = []
    lag_list_deg = []
    voltage_list = []
    h_list = []
    for ee in tracker.line.elements:
        if ee.__class__.__name__ == 'Cavity':
            eecp = ee.copy(_context=xo.ContextCpu())
            if ee.voltage != 0:
                freq_list.append(eecp.frequency)
                lag_list_deg.append(eecp.lag)
                voltage_list.append(eecp.voltage)
                h_list.append(eecp.frequency*T_rev)

    tw = tracker.twiss(particle_ref=particle_ref, at_elements=[line.element_names[0]])

    dct={}
    dct['T_rev'] = T_rev
    dct['freq_list'] = freq_list
    dct['lag_list_deg'] = lag_list_deg
    dct['voltage_list'] = voltage_list
    dct['h_list'] = h_list
    dct['momentum_compaction_factor'] = tw['momentum_compaction_factor']
    dct['slip_factor'] = tw['slip_factor']
    return dct

def generate_longitudinal_coordinates(
                                    tracker=None,
                                    particle_ref=None,
                                    mass0=None, q0=None, gamma0=None,
                                    num_particles=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    p_increment=0.,
                                    distribution='gaussian',
                                    sigma_z=None
                                    ):

    if tracker is not None:
        assert particle_ref is not None
        dct = _characterize_tracker(tracker, particle_ref)

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
        assert tracker is not None
        circumference = tracker.line.get_length()

    if momentum_compaction_factor is None:
        assert tracker is not None
        momentum_compaction_factor = dct['momentum_compaction_factor']

    if rf_harmonic is None:
        assert tracker is not None
        rf_harmonic=dct['h_list']

    if rf_voltage is None:
        assert tracker is not None
        rf_voltage=dct['voltage_list']

    if rf_phase is None:
        assert tracker is not None
        rf_phase=(np.array(dct['lag_list_deg']) - 180)/180*np.pi

    if distribution != 'gaussian':
        raise NotImplementedError

    assert sigma_z is not None
    rfbucket = RFBucket(circumference=circumference,
                        gamma=gamma0,
                        mass_kg=mass0/(clight**2)*qe,
                        charge_coulomb=np.abs(q0)*qe,
                        alpha_array=np.atleast_1d(momentum_compaction_factor),
                        harmonic_list=np.atleast_1d(rf_harmonic),
                        voltage_list=np.atleast_1d(rf_voltage),
                        phi_offset_list=np.atleast_1d(rf_phase),
                        p_increment=p_increment)

    if sigma_z < 0.1 * circumference/np.max(np.atleast_1d(rf_harmonic)):
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

    return z_particles, delta_particles
