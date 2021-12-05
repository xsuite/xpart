import logging

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

from .rfbucket_matching import RFBucketMatcher
from .rfbucket_matching import ThermalDistribution
from .rf_bucket import RFBucket

logger = logging.getLogger(__name__)

def _characterize_tracker(tracker, particle_ref):

    if tracker.iscollective:
        logger.warning('Ignoring collective elements in particles generation.')
        tracker = tracker._supertracker

    line = tracker.line
    T_rev = line.get_length()/(particle_ref.beta0[0]*clight)
    freq_list = []
    lag_list_deg = []
    voltage_list = []
    h_list = []
    for ee in tracker.line.elements:
        if ee.__class__.__name__ == 'Cavity':
            if ee.voltage != 0:
                freq_list.append(ee.frequency)
                lag_list_deg.append(ee.lag)
                voltage_list.append(ee.voltage)
                h_list.append(ee.frequency*T_rev)

    particle_co = tracker.find_closed_orbit(particle_ref)

    R_matrix = tracker.compute_one_turn_matrix_finite_differences(
                       particle_on_co=particle_co)

    eta = -R_matrix[4, 5]/line.get_length() # minus sign comes from z = s-ct
    alpha_mom_compaction = eta + 1/particle_ref.gamma0[0]**2

    dct={}
    dct['T_rev'] = T_rev
    dct['freq_list'] = freq_list
    dct['lag_list_deg'] = lag_list_deg
    dct['voltage_list'] = voltage_list
    dct['h_list'] = h_list
    dct['alpha_momentum_compaction'] = alpha_mom_compaction
    dct['eta'] = eta
    return dct

def generate_longitudinal_coordinates(
                                    tracker=None,
                                    particle_ref=None,
                                    mass0=None, q0=None, gamma0=None,
                                    num_particles=None,
                                    circumference=None,
                                    alpha_momentum_compaction=None,
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
        gamma0 = particle_ref.gamma0[0]

    if circumference is None:
        assert tracker is not None
        circumference = tracker.line.get_length()

    if alpha_momentum_compaction is None:
        assert tracker is not None
        alpha_momentum_compaction = dct['alpha_momentum_compaction']

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
                           charge_coulomb=q0*qe,
                           alpha_array=np.atleast_1d(alpha_momentum_compaction),
                           harmonic_list=np.atleast_1d(rf_harmonic),
                           voltage_list=np.atleast_1d(rf_voltage),
                           phi_offset_list=np.atleast_1d(rf_phase),
                           p_increment=p_increment)

    # Generate longitudinal coordinates
    matcher = RFBucketMatcher(rfbucket=rfbucket,
        distribution_type=ThermalDistribution,
        sigma_z=sigma_z)
    z_particles, delta_particles, _, _ = matcher.generate(
                                             macroparticlenumber=num_particles)

    return z_particles, delta_particles
