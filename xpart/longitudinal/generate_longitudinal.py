import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

from .rfbucket_matching import RFBucketMatcher
from .rfbucket_matching import ThermalDistribution
from .rf_bucket import RFBucket

def generate_longitudinal_coordinates(
                                    mass0, q0, gamma0,
                                    num_particles,
                                    circumference,
                                    alpha_momentum_compaction,
                                    rf_harmonic,
                                    rf_voltage,
                                    rf_phase,
                                    p_increment=0.,
                                    distribution='gaussian',
                                    sigma_z=None
                                    ):
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
